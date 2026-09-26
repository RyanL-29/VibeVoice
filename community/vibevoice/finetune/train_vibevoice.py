# train_vibevoice_lora.py
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, DatasetDict, VerificationMode, load_dataset, load_from_disk

from transformers import (
    HfArgumentParser,
    Trainer,
    set_seed,
    TrainerCallback,
)
from transformers import TrainingArguments as HfTrainingArguments

from peft import LoraConfig, get_peft_model, TaskType, set_peft_model_state_dict

from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration
from vibevoice.modular.configuration_vibevoice import VibeVoiceConfig
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor

from vibevoice.finetune.data_vibevoice import (
    TARGET_WAVEFORM_SPEC, VOICE_PROMPT_SOURCES, VibeVoiceCollator, VibeVoiceDataset,
)

logger = logging.getLogger(__name__)

# Collator workers fork after the fast tokenizer has been used; silence the HF tokenizers fork warning.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# File names inside <checkpoint>/lora/ (kept compatible with merge_vibevoice_models.py and lora_loading.py)
LORA_SUBDIR = "lora"
DIFFUSION_HEAD_DIR = "diffusion_head"            # save_pretrained() of the head -> model.safetensors (+config)
DIFFUSION_HEAD_FULL = "diffusion_head_full.bin"  # full head state_dict (lora_loading fallback)
DIFFUSION_HEAD_TRAINED = "diffusion_head_trained.bin"  # raw trained head, always (used for resume)
DIFFUSION_HEAD_EMA = "diffusion_head_ema.bin"    # EMA of the trainable head params (never swapped into the model)
CONNECTOR_WEIGHTS = "pytorch_model.bin"          # <lora>/{acoustic,semantic}_connector/pytorch_model.bin
ADAPTER_SAFE = "adapter_model.safetensors"
ADAPTER_BIN = "adapter_model.bin"


# ================== EMA ==================

class HeadEMA:
    """EMA of the diffusion head's trainable parameters.

    * Shadow is kept in fp32 on the same device as the parameters (bf16 shadows lose small updates to rounding).
    * Updated with a single fused ``torch._foreach_lerp_`` per optimizer step.
    * Decay ramps up as ``min(decay, (1 + step) / (10 + step))`` so early steps are not dominated by the base weights.
    * The shadow is NEVER swapped into the live model: it is saved separately (``diffusion_head_ema.bin``) and only
      exported as the inference head when ``--export_ema_head`` is set.
    """

    def __init__(self, decay: float = 0.999):
        self.decay = float(decay)
        self.names: List[str] = []
        self.shadow: Optional[List[torch.Tensor]] = None
        self.num_updates = 0

    @staticmethod
    def _trainable(head: nn.Module):
        return [(n, p) for n, p in head.named_parameters() if p.requires_grad]

    @torch.no_grad()
    def init_from(self, head: nn.Module) -> None:
        named = self._trainable(head)
        self.names = [n for n, _ in named]
        self.shadow = [p.detach().float().clone() for _, p in named]
        self.num_updates = 0

    @torch.no_grad()
    def update(self, head: nn.Module, step: int) -> None:
        if self.shadow is None:
            self.init_from(head)
            return
        params = [p for _, p in self._trainable(head)]
        if len(params) != len(self.shadow):
            raise ValueError(f"EMA tracks {len(self.shadow)} tensors but head now has {len(params)} trainable params")
        decay = min(self.decay, (1.0 + step) / (10.0 + step))
        live = [p if p.dtype == torch.float32 else p.float() for p in params]
        torch._foreach_lerp_(self.shadow, live, 1.0 - decay)
        self.num_updates += 1

    def state_dict(self) -> Dict[str, Any]:
        if self.shadow is None:
            return {}
        return {
            "decay": self.decay,
            "num_updates": self.num_updates,
            "params": {n: t.detach().cpu() for n, t in zip(self.names, self.shadow)},
        }

    @torch.no_grad()
    def load_state_dict(self, sd: Dict[str, Any], head: nn.Module) -> None:
        named = dict(self._trainable(head))
        params = sd["params"]
        missing = [n for n in named if n not in params]
        if missing:
            raise ValueError(f"EMA state is missing {len(missing)} head params, e.g. {missing[:3]}")
        self.names = list(named.keys())
        self.shadow = [params[n].to(device=named[n].device, dtype=torch.float32).clone() for n in self.names]
        self.num_updates = int(sd.get("num_updates", 0))

    def head_state_dict(self, head: nn.Module) -> Dict[str, torch.Tensor]:
        """Full head state_dict with the trainable params replaced by their EMA (dtype of the live head)."""
        sd = {k: v.detach() for k, v in head.state_dict().items()}
        for n, t in zip(self.names, self.shadow or []):
            if n not in sd:
                raise ValueError(f"EMA param {n} not found in head state_dict")
            sd[n] = t.to(dtype=sd[n].dtype, device=sd[n].device)
        return sd


class EmaCallback(TrainerCallback):
    def __init__(self, ema: HeadEMA):
        self.ema = ema

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        # When resuming, _load_from_checkpoint() has already restored the shadow; don't overwrite it.
        if self.ema.shadow is None and model is not None:
            self.ema.init_from(model.model.prediction_head)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self.ema.update(model.model.prediction_head, int(state.global_step))

class LoRADebugCallback(TrainerCallback):
    """Checks that LoRA A/B tensors are actually changing. Runs only at train start and on logging steps,
    using one batched norm computation (a single device->host copy) instead of one sync per tensor."""

    def __init__(self, log_every_n_steps: int = 50):
        self.log_every_n_steps = max(1, int(log_every_n_steps))
        self.names: List[str] = []
        self.is_b: Optional[torch.Tensor] = None
        self.prev: Optional[torch.Tensor] = None

    def _norms(self, model) -> torch.Tensor:
        named = dict(model.named_parameters())
        norms = torch._foreach_norm([named[n].detach() for n in self.names])
        return torch.stack([t.float() for t in norms]).cpu()

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is None:
            return
        named = dict(model.named_parameters())
        self.names = [n for n in named if ("lora_A" in n or "lora_B" in n)]
        if not self.names:
            logger.warning("LoRA debug: No LoRA parameters found. Check lora_target_modules.")
            return
        self.is_b = torch.tensor(["lora_B" in n for n in self.names])
        self.prev = self._norms(model)
        req_grad = sum(1 for n in self.names if named[n].requires_grad)
        logger.info(
            f"LoRA debug: found {len(self.names)} LoRA params (A={int((~self.is_b).sum())}, B={int(self.is_b.sum())}); "
            f"trainable={req_grad}. Initial lora_B_zero={int(((self.prev == 0) & self.is_b).sum())}."
        )
        if req_grad != len(self.names):
            logger.warning("LoRA debug: Some LoRA params are frozen. They should be trainable.")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if model is None or self.prev is None:
            return
        step = int(state.global_step or 0)
        if step % self.log_every_n_steps != 0 and step != 1:
            return
        curr = self._norms(model)
        changed = (curr - self.prev).abs() > 1e-12
        b = self.is_b
        logger.info(
            f"LoRA debug step {step}: changed A {int((changed & ~b).sum())}/{int((~b).sum())}, "
            f"changed B {int((changed & b).sum())}/{int(b.sum())}, lora_B_zero_now={int(((curr == 0) & b).sum())}."
        )
        self.prev = curr


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to VibeVoice base model with config.json"}
    )
    processor_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Path to processor dir (preprocessor_config.json). Defaults to model path."}
    )
    cache_dir: Optional[str] = field(default=None)
    freeze_acoustic_tokenizer: bool = field(default=True)
    freeze_semantic_tokenizer: bool = field(default=True)
    lora_r: int = field(default=8)
    lora_alpha: int = field(default=32)
    lora_dropout: float = field(default=0.05)
    lora_target_modules: str = field(
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        metadata={"help": "Comma-separated list of target module names in the LLM blocks"},
    )
    lora_wrap_diffusion_head: bool = field(default=False, metadata={"help": "Wrap diffusion head with PEFT LoRA"})
    train_diffusion_head: bool = field(default=False, metadata={"help": "Train diffusion prediction head (full fine-tune)"})
    train_connectors: bool = field(default=False, metadata={"help": "Train acoustic/semantic connectors (full fine-tune)"})
    layers_to_freeze: Optional[str] = field(
        default=None, 
        metadata={"help": "Comma-separated indices of diffusion head layers to freeze (e.g., '0,1,5,7,8')."}
    )
    attn_implementation: str = field(
        default="sdpa",
        metadata={"help": "Attention kernel for the Qwen2 LM: 'sdpa' (default), 'eager', or 'flash_attention_2'."},
    )

@dataclass
class DataArguments:
    dataset_name: Optional[str] = field(default=None, metadata={"help": "Path to a dataset saved with save_to_disk (Dataset or DatasetDict)"})
    dataset_config_name: Optional[str] = field(default=None)
    train_split_name: str = field(default="train")
    eval_split_name: Optional[str] = field(default="validation")
    text_column_name: str = field(default="text")
    audio_column_name: str = field(default="audio")
    voice_prompts_column_name: Optional[str] = field(default="voice_prompts")
    voice_prompt_source: str = field(
        default="other",
        metadata={"help": "Where voice prompts come from: 'other' (random crop of a DIFFERENT clip, default; assumes a "
                          "single-speaker dataset), 'self_crop' (crop of the target clip itself, legacy), or 'column' "
                          "(use --voice_prompts_column_name, falling back to self_crop when missing)."},
    )
    voice_prompt_max_sec: float = field(default=10.0, metadata={"help": "Max voice prompt length (s) for 'other'."})
    voice_prompt_min_sec: float = field(default=3.0, metadata={"help": "Min voice prompt length (s) for 'other'."})
    eval_split_size: float = field(default=0.0)
    ignore_verifications: bool = field(default=False)
    max_length: Optional[int] = field(default=None)
    train_jsonl: Optional[str] = field(default=None, metadata={"help": "Path to local train JSONL with {text, audio, [voice_prompts]}"})
    validation_jsonl: Optional[str] = field(default=None, metadata={"help": "Optional path to local validation JSONL"})
    voice_prompt_drop_rate: float = field(
        default=0.0,
        metadata={"help": "Probability to drop conditioning voice prompt during training (0.0 keep always, 1.0 drop always)."},
    )


@dataclass
class CustomTrainingArguments(HfTrainingArguments):
    ddpm_batch_mul: int = field(default=1)
    ce_loss_weight: float = field(default=1.0)
    diffusion_loss_weight: float = field(default=1.0)
    ce_on_speech_control_only: bool = field(
        default=False,
        metadata={"help": "Restrict CE loss to the speech control tokens (speech_end / eos) instead of every "
                          "non-acoustic token (system prompt, text, ...)."},
    )
    debug_ce_details: bool = field(default=False)
    debug_ce_topk: int = field(default=5)
    debug_ce_max_examples: int = field(default=1)
    debug_ce_every_n_steps: int = field(default=200)
    debug_checks: bool = field(
        default=False,
        metadata={"help": "Extra (GPU-syncing) consistency checks in the collator, and the startup CE smoke test."},
    )
    gradient_clipping: bool = field(
        default=False,
        metadata={"help": "Enable gradient clipping using max_grad_norm (set via --max_grad_norm, default 1.0). When False, disables clipping by forcing max_grad_norm=0.0."},
    )
    debug_save: bool = field(
        default=False,
        metadata={"help": "If set, saves model components BEFORE training starts, into output_dir/debug_initial."},
    )
    cache_target_features: bool = field(
        default=True,
        metadata={"help": "Cache the frozen acoustic/semantic encoder outputs of each training target clip on the GPU "
                          "(~25 KB/clip) so epochs 2+ skip re-encoding. Acoustic sampling noise is still fresh each step."},
    )
    target_feature_cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Directory for offline-precomputed target features. Before training, every training clip is "
                          "encoded once (acoustic + semantic) into <dir>/target_features-<hash>.safetensors, and later "
                          "runs with the same dataset/model/audio pipeline reuse it, so no epoch re-encodes targets. "
                          "Requires --cache_target_features True. Voice prompts are still encoded on the fly."},
    )
    use_ema: bool = field(default=True, metadata={"help": "Track an EMA of the diffusion head (saved as diffusion_head_ema.bin)."})
    ema_decay: float = field(default=0.999, metadata={"help": "EMA decay (ramped as min(decay, (1+step)/(10+step)))."})
    export_ema_head: bool = field(
        default=False,
        metadata={"help": "Export the EMA head (instead of the raw trained head) as the inference head "
                          "(diffusion_head/model.safetensors + diffusion_head_full.bin). Requires --use_ema."},
    )

def build_lora_config(args: ModelArguments) -> LoraConfig:
    target_modules = [s.strip() for s in args.lora_target_modules.split(",") if s.strip()]
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=target_modules,
    )

def build_head_lora_config(args: ModelArguments) -> LoraConfig:
    target_modules = ["noisy_images_proj","cond_proj","gate_proj","up_proj","down_proj","linear"]
    return LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        target_modules=target_modules,
    )

def mask_for_ce(labels: torch.Tensor, attention_mask: torch.Tensor, acoustic_input_mask: torch.Tensor, pad_id: int = -100,
                keep_token_ids: Optional[torch.Tensor] = None,
                extra_keep_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Next-token CE labels (shifted by one).

    Default: score every non-padding target except acoustic placeholders.
    If keep_token_ids is given ("speech control only"): score only targets whose id is in keep_token_ids, plus the
    positions in extra_keep_mask (used for the *target* speech placeholders, so the model also sees "continue" examples
    and does not simply learn to always emit speech_end)."""
    shifted = labels[:, 1:].contiguous()
    base_mask = attention_mask[:, 1:].contiguous().eq(1) if (attention_mask is not None and attention_mask.numel() > 0) else torch.ones_like(shifted, dtype=torch.bool)
    if keep_token_ids is None:
        label_is_acoustic = acoustic_input_mask[:, 1:].contiguous()
        final_mask = base_mask & (~label_is_acoustic)
    else:
        keep = torch.isin(shifted, keep_token_ids.to(shifted.device))
        if extra_keep_mask is not None:
            keep |= extra_keep_mask[:, 1:]
        final_mask = base_mask & keep
    out = shifted.clone()
    out[~final_mask] = pad_id
    return out

def _patch_acoustic_encode_for_legacy_indexing(model_obj, logger_):
    """model.forward_speech_features() indexes encode(...)[0][0]; the tokenizer returns a dataclass. Wrap it."""
    try:
        acoustic = getattr(getattr(model_obj, "model", model_obj), "acoustic_tokenizer", None)
        if acoustic is None or not hasattr(acoustic, "encode"):
            logger_.warning("No acoustic_tokenizer.encode() found to patch.")
            return
        base_encode = acoustic.encode
        def encode_wrapped(*args, **kwargs):
            out = base_encode(*args, **kwargs)
            try:
                _ = out[0][0]
                return out
            except Exception:
                pass
            if isinstance(out, dict):
                for k in ("frames", "codes", "tokens", "latents", "hidden_states"):
                    if k in out:
                        return [[out[k]]]
                if len(out) > 0:
                    return [[next(iter(out.values()))]]
            for attr in ("frames", "codes", "tokens", "latents", "hidden_states"):
                if hasattr(out, attr):
                    return [[getattr(out, attr)]]
            return [[out]]
        acoustic.encode = encode_wrapped
        logger_.info("Patched acoustic_tokenizer.encode() to return [[...]] for legacy indexing.")
    except Exception as e:
        logger_.warning(f"Failed to patch acoustic_tokenizer.encode(): {e}")


def _is_peft(module) -> bool:
    return module is not None and hasattr(module, "peft_config") and hasattr(module, "save_pretrained")


# ================== SAVE / LOAD ==================

def save_vibevoice_assets(model: VibeVoiceForConditionalGeneration, output_dir: str,
                          ema: Optional[HeadEMA] = None, export_ema_head: bool = False) -> str:
    """Write everything needed for inference/merging/resume into <output_dir>/lora/:

      adapter_model.safetensors + adapter_config.json   LLM LoRA (if the LLM is PEFT-wrapped)
      diffusion_head/                                   head: save_pretrained() (model.safetensors+config) or PEFT adapter
      diffusion_head/diffusion_head_full.bin            full head state_dict (the *exported* head)
      diffusion_head_trained.bin                        raw trained head, only when the export is the EMA head
      diffusion_head_ema.bin                            EMA of trainable head params (fp32), if EMA is enabled
      {acoustic,semantic}_connector/pytorch_model.bin   connectors

    Readers: scripts/merge_vibevoice_models.py and modular/lora_loading.py (exported head), _load_from_checkpoint (resume).
    Nothing is swapped into the live model, so saving never changes training weights.
    """
    lora_out = os.path.join(output_dir, LORA_SUBDIR)
    ph_dir = os.path.join(lora_out, DIFFUSION_HEAD_DIR)
    os.makedirs(ph_dir, exist_ok=True)
    inner = model.model

    lm = getattr(inner, "language_model", None)
    if _is_peft(lm):
        lm.save_pretrained(lora_out)

    def cpu_sd(sd):  # CPU copies: .bin files load anywhere without map_location, and the live tensors are untouched
        return {k: v.detach().to("cpu", copy=True) for k, v in sd.items()}

    ph = getattr(inner, "prediction_head", None)
    if ph is not None:
        use_ema = export_ema_head and ema is not None and ema.shadow is not None
        trained_sd = cpu_sd(ph.state_dict())
        export_sd = cpu_sd(ema.head_state_dict(ph)) if use_ema else trained_sd
        # save_pretrained takes the state_dict explicitly, so the live head is untouched
        ph.save_pretrained(ph_dir, state_dict=export_sd)
        torch.save(export_sd, os.path.join(ph_dir, DIFFUSION_HEAD_FULL))
        stale_trained = os.path.join(lora_out, DIFFUSION_HEAD_TRAINED)
        if use_ema:
            torch.save(trained_sd, stale_trained)
        elif os.path.exists(stale_trained):
            os.remove(stale_trained)
        if ema is not None and ema.shadow is not None:
            torch.save(ema.state_dict(), os.path.join(lora_out, DIFFUSION_HEAD_EMA))

    for name in ("acoustic_connector", "semantic_connector"):
        conn = getattr(inner, name, None)
        if conn is not None:
            d = os.path.join(lora_out, name)
            os.makedirs(d, exist_ok=True)
            torch.save(cpu_sd(conn.state_dict()), os.path.join(d, CONNECTOR_WEIGHTS))
    return lora_out


def _load_adapter_weights(peft_module, adapter_dir: str) -> bool:
    safe_f, bin_f = os.path.join(adapter_dir, ADAPTER_SAFE), os.path.join(adapter_dir, ADAPTER_BIN)
    if os.path.isfile(safe_f):
        from safetensors.torch import load_file
        sd = load_file(safe_f, device="cpu")
    elif os.path.isfile(bin_f):
        sd = torch.load(bin_f, map_location="cpu")
    else:
        return False
    res = set_peft_model_state_dict(peft_module, sd)
    unexpected = getattr(res, "unexpected_keys", None) or []
    if unexpected:
        raise ValueError(f"Adapter in {adapter_dir} has {len(unexpected)} unexpected keys, e.g. {unexpected[:3]}")
    return True


def load_vibevoice_assets(model: VibeVoiceForConditionalGeneration, checkpoint_dir: str,
                          ema: Optional[HeadEMA] = None) -> List[str]:
    """Inverse of save_vibevoice_assets(), used for --resume_from_checkpoint. Returns the restored components."""
    lora_dir = os.path.join(checkpoint_dir, LORA_SUBDIR)
    if not os.path.isdir(lora_dir):
        raise ValueError(f"{checkpoint_dir} has no '{LORA_SUBDIR}/' directory; not a VibeVoice fine-tune checkpoint")
    inner, restored = model.model, []
    ph_dir = os.path.join(lora_dir, DIFFUSION_HEAD_DIR)

    lm = getattr(inner, "language_model", None)
    if _is_peft(lm):
        if not _load_adapter_weights(lm, lora_dir):
            raise ValueError(f"LLM is LoRA-wrapped but no adapter weights found in {lora_dir}")
        restored.append("llm_lora")

    ph = getattr(inner, "prediction_head", None)
    if _is_peft(ph):
        if not _load_adapter_weights(ph, ph_dir):
            raise ValueError(f"Diffusion head is LoRA-wrapped but no adapter found in {ph_dir}")
        restored.append("diffusion_head_lora")
    elif ph is not None:
        # Prefer the raw trained head; older checkpoints only have the exported one.
        candidates = [os.path.join(lora_dir, DIFFUSION_HEAD_TRAINED), os.path.join(ph_dir, DIFFUSION_HEAD_FULL),
                      os.path.join(lora_dir, DIFFUSION_HEAD_FULL), os.path.join(ph_dir, "model.safetensors")]
        path = next((p for p in candidates if os.path.isfile(p)), None)
        if path is None:
            raise ValueError(f"No diffusion head weights in {lora_dir}")
        if path.endswith(".safetensors"):
            from safetensors.torch import load_file
            sd = load_file(path, device="cpu")
        else:
            sd = torch.load(path, map_location="cpu")
        ph.load_state_dict(sd, strict=True)
        restored.append(f"diffusion_head({os.path.basename(path)})")

    for name in ("acoustic_connector", "semantic_connector"):
        conn, p = getattr(inner, name, None), os.path.join(lora_dir, name, CONNECTOR_WEIGHTS)
        if conn is not None and os.path.isfile(p):
            conn.load_state_dict(torch.load(p, map_location="cpu"), strict=True)
            restored.append(name)

    ema_path = os.path.join(lora_dir, DIFFUSION_HEAD_EMA)
    if ema is not None and ph is not None:
        if os.path.isfile(ema_path):
            ema.load_state_dict(torch.load(ema_path, map_location="cpu"), ph)
            restored.append("ema")
        else:
            logger.warning(f"No {DIFFUSION_HEAD_EMA} in checkpoint; EMA restarts from the resumed head.")
    return restored


# ================== TARGET FEATURE PRECOMPUTE ==================

FEATURE_CACHE_FORMAT = 1
FEATURE_KINDS = (("acoustic", "acoustic_tokenizer"), ("semantic", "semantic_tokenizer"))


def _encoder_mean(tokenizer, wav: torch.Tensor) -> torch.Tensor:
    """[B, samples] waveform -> [B, frames, D] encoder mean of a frozen speech tokenizer (computed in its dtype)."""
    dtype = next(tokenizer.parameters()).dtype
    out = tokenizer.encode(wav.to(dtype=dtype).unsqueeze(1))
    if isinstance(out, (list, tuple)):  # legacy [[output]] wrapper, see _patch_acoustic_encode_for_legacy_indexing
        out = out[0][0]
    return out.mean


class _TargetWaveforms(torch.utils.data.Dataset):
    """(index, target waveform exactly as the collator builds it); lets DataLoader workers decode audio in parallel."""

    def __init__(self, dataset: VibeVoiceDataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        return idx, torch.from_numpy(self.dataset.target_waveform(idx))


@torch.no_grad()
def _tokenizer_fingerprint(tokenizer) -> Dict[str, Any]:
    """Cheap identity of a frozen tokenizer's encoder: config, dtype and a per-tensor weight checksum."""
    enc = getattr(tokenizer, "encoder", tokenizer)
    params = list(enc.parameters())
    sums = torch.stack([p.detach().double().sum() for p in params]).cpu().tolist()
    return {
        "config": tokenizer.config.to_dict() if hasattr(tokenizer, "config") else None,
        "dtype": str(params[0].dtype),
        "shapes": [list(p.shape) for p in params],
        "checksum": [float(f"{s:.9g}") for s in sums],
    }


def target_feature_cache_path(cache_dir: str, model: VibeVoiceForConditionalGeneration, train_ds,
                              audio_column: str) -> str:
    """<cache_dir>/target_features-<hash>.safetensors; the hash covers everything the cached features depend on."""
    import hashlib
    import json

    fingerprint = getattr(train_ds, "_fingerprint", None)
    if not fingerprint:
        raise ValueError("--target_feature_cache_dir needs a datasets.Dataset with a fingerprint "
                         f"(got {type(train_ds).__name__}); drop the flag to encode on the fly.")
    ident = {
        "format": FEATURE_CACHE_FORMAT,
        "waveform": TARGET_WAVEFORM_SPEC,
        "dataset": {"fingerprint": fingerprint, "rows": len(train_ds), "audio_column": audio_column},
        "tokenizers": {kind: _tokenizer_fingerprint(getattr(model.model, attr)) for kind, attr in FEATURE_KINDS},
    }
    digest = hashlib.sha256(json.dumps(ident, sort_keys=True, default=str).encode()).hexdigest()[:16]
    return os.path.join(cache_dir, f"target_features-{digest}.safetensors")


@torch.no_grad()
def precompute_target_features(model: VibeVoiceForConditionalGeneration, dataset: VibeVoiceDataset, path: str,
                               num_workers: int = 0) -> None:
    """Encode every training target clip once with the frozen acoustic + semantic encoders and write <path>
    (safetensors): {acoustic,semantic}: [sum_frames, D] means concatenated over clips, frames: [N] frames per clip,
    num_samples: [N] waveform length per clip (re-checked against every batch at train time).

    Each clip is encoded alone at its exact length, exactly as VibeVoiceTrainer._encode_segments does (zero-padded
    batches change the outputs), so the cached features are bit-identical to the on-the-fly ones."""
    import time
    from safetensors.torch import save_file

    toks = {kind: getattr(model.model, attr) for kind, attr in FEATURE_KINDS}
    for t in toks.values():
        t.eval()
    device = next(toks["acoustic"].parameters()).device
    loader = torch.utils.data.DataLoader(
        _TargetWaveforms(dataset), batch_size=None, shuffle=False, num_workers=num_workers,
        pin_memory=device.type == "cuda", prefetch_factor=8 if num_workers > 0 else None,
    )
    n = len(dataset)
    chunks: Dict[str, List[torch.Tensor]] = {kind: [] for kind in toks}
    frames: List[int] = []
    num_samples: List[int] = []
    start = last_log = time.time()
    for i, (idx, wav) in enumerate(loader):
        if int(idx) != i:
            raise RuntimeError(f"precompute loader returned index {int(idx)} at position {i}")
        wav = wav.to(device, non_blocking=True)[None]
        for kind, tok in toks.items():
            chunks[kind].append(_encoder_mean(tok, wav)[0].to("cpu", non_blocking=False))
        frames.append(int(chunks["acoustic"][-1].shape[0]))
        num_samples.append(int(wav.shape[-1]))
        if chunks["semantic"][-1].shape[0] != frames[-1]:
            raise RuntimeError(f"clip {i}: acoustic/semantic frame counts differ")
        now = time.time()
        if now - last_log > 30 or i + 1 == n:
            rate = (i + 1) / max(now - start, 1e-6)
            logger.info(f"Precomputing target features: {i + 1}/{n} clips ({rate:.1f} clips/s, "
                        f"ETA {(n - i - 1) / rate / 60:.1f} min)")
            last_log = now
    if len(frames) != n:
        raise RuntimeError(f"precompute produced {len(frames)} clips, expected {n}")
    tensors = {kind: torch.cat(c).contiguous() for kind, c in chunks.items()}
    tensors["frames"] = torch.tensor(frames, dtype=torch.int64)
    tensors["num_samples"] = torch.tensor(num_samples, dtype=torch.int64)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp-{os.getpid()}"
    save_file(tensors, tmp, metadata={"format": str(FEATURE_CACHE_FORMAT), "rows": str(n)})
    os.replace(tmp, path)  # atomic: an interrupted precompute never leaves a truncated cache file behind
    logger.info(f"Saved target features for {n} clips to {path} "
                f"({os.path.getsize(path) / 2**20:.0f} MB, {time.time() - start:.0f} s)")


def load_target_features(path: str, num_rows: int, device: torch.device) -> Dict[Any, Any]:
    """Read a precompute file -> {(kind, index): (num_samples, mean view on `device`)} for VibeVoiceTrainer."""
    from safetensors.torch import load_file

    data = load_file(path, device="cpu")
    frames, num_samples = data["frames"].tolist(), data["num_samples"].tolist()
    if len(frames) != num_rows or len(num_samples) != num_rows:
        raise ValueError(f"{path} has {len(frames)} clips but the train dataset has {num_rows}; delete it to rebuild")
    out: Dict[Any, Any] = {}
    for kind, _ in FEATURE_KINDS:
        flat = data[kind]
        if flat.shape[0] != sum(frames):
            raise ValueError(f"{path}: '{kind}' has {flat.shape[0]} frames, expected {sum(frames)}; delete it to rebuild")
        # a single device copy per kind; per-clip entries are views into it
        for idx, (ns, m) in enumerate(zip(num_samples, torch.split(flat.to(device), frames))):
            out[(kind, idx)] = (int(ns), m)
    return out


def setup_target_feature_cache(trainer: "VibeVoiceTrainer", model: VibeVoiceForConditionalGeneration,
                               train_dataset: VibeVoiceDataset, train_ds, data_args: DataArguments,
                               training_args: "CustomTrainingArguments") -> None:
    """Load the precomputed target features, computing and saving them first if the file does not exist yet."""
    if not training_args.cache_target_features:
        raise ValueError("--target_feature_cache_dir requires --cache_target_features True")
    path = target_feature_cache_path(training_args.target_feature_cache_dir, model, train_ds,
                                     data_args.audio_column_name)
    # With several processes the main one builds the file while the others wait, then every rank loads it.
    with training_args.main_process_first(desc="precompute target features"):
        if os.path.isfile(path):
            logger.info(f"Using precomputed target features {path}")
        else:
            logger.info(f"No precomputed target features at {path}; encoding {len(train_dataset)} clips once "
                        "(later runs on the same dataset/model reuse the file)")
            # The DataLoader draws worker seeds from the global RNG; fork it so building the file does not change the
            # training run's random stream (a run that builds the cache == a run that reuses it).
            with torch.random.fork_rng(devices=[]):
                state = random.getstate()
                try:
                    precompute_target_features(model, train_dataset, path,
                                               num_workers=training_args.dataloader_num_workers)
                finally:
                    random.setstate(state)
    device = next(model.model.acoustic_tokenizer.parameters()).device
    trainer.load_precomputed_features(load_target_features(path, len(train_dataset), device))
    logger.info(f"Loaded precomputed features for {len(train_dataset)} target clips onto {device}")


# ================== TRAINER ==================

class VibeVoiceTrainer(Trainer):
    def __init__(self, *args, ema: Optional[HeadEMA] = None, ce_keep_token_ids: Optional[List[int]] = None,
                 cache_target_features: bool = True, **kwargs):
        super().__init__(*args, **kwargs)
        # (tokenizer, train dataset index) -> (num_samples, encoder mean on the GPU) (~25 KB per clip); filled lazily
        # during epoch 1, or up front by load_precomputed_features() (--target_feature_cache_dir).
        self._feature_cache: Optional[Dict[Any, Any]] = {} if cache_target_features else None
        # The model's forward(**kwargs) makes Trainer assume the model normalises the loss by num_items_in_batch,
        # so it would skip `loss / gradient_accumulation_steps` -> grads (and logged loss) GAS x too large.
        # Our compute_loss returns a per-micro-batch mean, so Trainer must do the GAS division.
        self.model_accepts_loss_kwargs = False
        self.ema = ema
        self._ce_keep_ids = torch.tensor(ce_keep_token_ids, dtype=torch.long) if ce_keep_token_ids else None
        self._loss_acc: Dict[str, Dict[str, Any]] = {}
        inner = self.model.model
        self._speech_scaling_ready = not bool(
            torch.isnan(inner.speech_scaling_factor) | torch.isnan(inner.speech_bias_factor)
        )

    def load_precomputed_features(self, features: Dict[Any, Any]) -> None:
        """Seed the target-feature cache with offline-precomputed entries (see precompute_target_features)."""
        if self._feature_cache is None:
            raise ValueError("Precomputed target features require --cache_target_features True")
        self._feature_cache.update(features)

    # ---------- forward ----------
    def _encode_segments(self, tokenizer, speech_tensors: torch.Tensor, lengths: Optional[List[int]], rows: List[int],
                         max_frames: int, cache_keys: Optional[List[Any]] = None) -> torch.Tensor:
        """Run a (frozen) speech tokenizer's encoder -> [len(rows), max_frames, D] means (zero-padded frames).

        With `lengths`, each segment is encoded at its exact length: cheaper than the zero-padded batch (the conv
        encoder's cost scales with the padded length) and identical to inference, where padding never reaches the
        last frame. Without lengths it falls back to one padded batch.
        cache_keys (aligned with rows, None = don't cache): the encoder mean is deterministic (frozen, eval mode;
        acoustic sampling noise is added afterwards), so target clips are encoded once and reused in later epochs.
        Cache values are (num_samples, mean); a hit whose sample count differs from the batch raises (stale cache)."""
        if lengths is None:
            return _encoder_mean(tokenizer, speech_tensors[rows])[:, :max_frames]
        means = []
        for i, r in enumerate(rows):
            key = cache_keys[i] if (cache_keys is not None and self._feature_cache is not None) else None
            hit = self._feature_cache.get(key) if key is not None else None
            if hit is None:
                m = _encoder_mean(tokenizer, speech_tensors[r : r + 1, : lengths[r]])[0]
                if key is not None:
                    self._feature_cache[key] = (int(lengths[r]), m)
            else:
                n_samples, m = hit
                if n_samples != lengths[r]:  # host ints only, no GPU sync
                    raise ValueError(f"Feature cache entry {key} was computed from {n_samples} samples but the batch "
                                     f"has {lengths[r]}; the precomputed features do not match this dataset. "
                                     "Delete the --target_feature_cache_dir file or disable the cache.")
            means.append(m)
        out = means[0].new_zeros(len(rows), max_frames, means[0].shape[-1])
        for i, m in enumerate(means):
            n = min(m.shape[0], max_frames)
            out[i, :n] = m[:n]
        return out

    def _acoustic_features(self, model, speech_tensors: torch.Tensor, speech_masks: torch.Tensor,
                           lengths: Optional[List[int]] = None, cache_keys: Optional[List[Any]] = None):
        """Same maths as model.forward_speech_features(speech_type="audio", return_unmask=True) -> ([S,T,64] latents,
        [S,T,H] connector outputs), minus its per-call `if isnan(scaling_factor)` host sync (it is only used to
        initialise the scaling buffers when they are NaN)."""
        inner = model.model
        inner.acoustic_tokenizer.eval()
        if not self._speech_scaling_ready:
            out = model.forward_speech_features(speech_tensors=speech_tensors, speech_masks=speech_masks,
                                                speech_type="audio", return_unmask=True)
            self._speech_scaling_ready = not bool(torch.isnan(inner.speech_scaling_factor) | torch.isnan(inner.speech_bias_factor))
            return out
        tok = inner.acoustic_tokenizer
        with torch.no_grad():
            mean = self._encode_segments(tok, speech_tensors, lengths, list(range(speech_tensors.shape[0])),
                                         speech_masks.shape[1], cache_keys=cache_keys)
            # identical to VibeVoiceTokenizerEncoderOutput(mean, fix_std).sample(std_dist_type)[0]
            if tok.std_dist_type == "gaussian":
                std = torch.randn(mean.shape[0], 1, 1, device=mean.device, dtype=mean.dtype) * (tok.fix_std / 0.8)
                latents = mean + std * torch.randn_like(mean)
            elif tok.std_dist_type == "fix":
                latents = mean + tok.fix_std * torch.randn_like(mean)
            else:
                latents = mean
            features = (latents + inner.speech_bias_factor) * inner.speech_scaling_factor
        return features, inner.acoustic_connector(features)

    def _semantic_connect(self, model, speech_tensors: torch.Tensor, target_rows: List[int], like: torch.Tensor,
                          lengths: Optional[List[int]] = None, cache_keys: Optional[List[Any]] = None) -> Optional[torch.Tensor]:
        """Semantic connector embeddings for the target segments (zeros for voice-prompt segments), matching
        inference: generated frames are fed back as acoustic + semantic, prompt frames as acoustic only."""
        tok = getattr(model.model, "semantic_tokenizer", None)
        if tok is None:
            raise ValueError("model.model.semantic_tokenizer is required to build semantic features")
        if not target_rows:
            return None
        tok.eval()
        with torch.no_grad():
            sem = self._encode_segments(tok, speech_tensors, lengths, target_rows, like.shape[1],  # [n_target, T, 128]
                                        cache_keys=cache_keys)
        conn = model.model.semantic_connector(sem.to(dtype=like.dtype))
        sel = torch.tensor(target_rows, device=like.device)
        return torch.zeros_like(like).index_copy(0, sel, conn)

    def training_forward(self, model: VibeVoiceForConditionalGeneration, inputs: Dict[str, Any]):
        """Returns (ce_loss, diffusion_loss, n_target_latents)."""
        input_ids = inputs["input_ids"]
        attention_mask = inputs.get("attention_mask")
        speech_tensors = inputs.get("speech_tensors")
        speech_masks = inputs.get("speech_masks")
        speeches_loss_input = inputs.get("speeches_loss_input")
        acoustic_input_mask = inputs.get("acoustic_input_mask")
        acoustic_loss_mask = inputs.get("acoustic_loss_mask")
        ddpm_batch_mul = max(1, int(self.args.ddpm_batch_mul))
        inner = model.model

        x = model.get_input_embeddings()(input_ids)
        if x.is_leaf and x.requires_grad:
            # enable_input_require_grads() turns the (frozen) embedding output into a grad-requiring leaf;
            # clone so the in-place placeholder write below is legal.
            x = x.clone()

        speech_features = None
        if speech_tensors is not None:
            if speech_masks is None or speeches_loss_input is None or acoustic_input_mask is None:
                raise ValueError("speech_tensors given without speech_masks / speeches_loss_input / acoustic_input_mask")
            # [S, T, vae_dim] latents (no grad) and [S, T, H] acoustic connector outputs
            lengths = inputs.get("speech_lengths")
            if lengths is not None and len(lengths) != speech_tensors.shape[0]:
                raise ValueError(f"speech_lengths has {len(lengths)} entries for {speech_tensors.shape[0]} segments")
            target_rows = inputs.get("target_segment_rows")
            if target_rows is None:  # host-side list, so no GPU sync; falls back to the mask for custom collators
                target_rows = speeches_loss_input.any(dim=1).nonzero(as_tuple=True)[0].tolist()
            # Cache keys only for *training* target clips (voice prompts are random crops, eval uses another dataset)
            keys = inputs.get("target_keys") if model.training else None
            ac_keys = sem_keys = None
            if keys is not None and len(keys) == len(target_rows) and all(k is not None for k in keys):
                ac_keys = [None] * speech_tensors.shape[0]
                for r, k in zip(target_rows, keys):
                    ac_keys[r] = ("acoustic", k)
                sem_keys = [("semantic", k) for k in keys]
            speech_all_features, speech_all_connect = self._acoustic_features(
                model, speech_tensors.type_as(x), speech_masks, lengths, cache_keys=ac_keys)
            if speech_all_features.shape[:2] != speech_masks.shape:
                raise ValueError(f"acoustic latents {tuple(speech_all_features.shape)} vs speech_masks {tuple(speech_masks.shape)}")
            sem_add = self._semantic_connect(model, speech_tensors, target_rows, speech_all_connect, lengths, cache_keys=sem_keys)
            combined = speech_all_connect if sem_add is None else speech_all_connect + sem_add
            placeholders = combined[speech_masks]
            try:
                x[acoustic_input_mask] = placeholders
            except RuntimeError as e:
                raise ValueError(f"{placeholders.shape[0]} speech latents do not fit the acoustic_input_mask placeholders: {e}") from e
            speech_features = speech_all_features[speeches_loss_input & speech_masks]  # diffusion targets [N, vae_dim]

        outputs = inner(input_ids=None, attention_mask=attention_mask, inputs_embeds=x, use_cache=False, return_dict=True)
        hidden_states = outputs.last_hidden_state

        # ----- diffusion loss -----
        n_lat = 0 if speech_features is None else int(speech_features.shape[0])
        if n_lat > 0:
            # Condition on the hidden state of the position BEFORE each target latent (next-token alignment).
            cond_mask = torch.zeros_like(acoustic_loss_mask)
            cond_mask[:, :-1] = acoustic_loss_mask[:, 1:]
            condition = hidden_states[cond_mask]
            if condition.shape[0] != n_lat:
                raise ValueError(f"{condition.shape[0]} diffusion conditions vs {n_lat} target latents")
            latent_size = speech_features.shape[1]
            n = n_lat * ddpm_batch_mul
            noise = torch.randn((n, latent_size), device=hidden_states.device, dtype=hidden_states.dtype)
            timesteps = torch.randint(0, model.config.diffusion_head_config.ddpm_num_steps, (n,), device=hidden_states.device)
            target_latents = speech_features.repeat_interleave(ddpm_batch_mul, dim=0)
            noisy = inner.noise_scheduler.add_noise(target_latents, noise, timesteps)
            pred = inner.prediction_head(noisy, timesteps.type_as(x), condition.repeat_interleave(ddpm_batch_mul, dim=0))
            prediction_type = model.config.diffusion_head_config.prediction_type
            if prediction_type == "epsilon":
                target = noise
            elif prediction_type == "v_prediction":
                target = inner.noise_scheduler.get_velocity(target_latents, noise, timesteps)
            else:
                raise NotImplementedError(f"Prediction type {prediction_type} not implemented")
            diffusion_loss = F.mse_loss(pred.float(), target.float(), reduction="mean")
        else:
            # keep every trainable speech module in the graph (DDP) with a zero contribution
            diffusion_loss = hidden_states.sum() * 0.0
            for mod in (inner.prediction_head, inner.acoustic_connector, inner.semantic_connector):
                diffusion_loss = diffusion_loss + sum(p.sum() for p in mod.parameters() if p.requires_grad) * 0.0

        # ----- CE loss (logits only where a label is scored) -----
        ce_labels = mask_for_ce(
            input_ids, attention_mask, acoustic_input_mask, pad_id=-100,
            keep_token_ids=self._ce_keep_ids, extra_keep_mask=acoustic_loss_mask if self._ce_keep_ids is not None else None,
        )
        valid = ce_labels.ne(-100)
        h = hidden_states[:, :-1][valid]
        if h.shape[0] > 0:
            logits = model.lm_head(h).float()
            labels = ce_labels[valid]
            ce_loss = F.cross_entropy(logits, labels)
            self._debug_ce(logits, labels, valid)
        else:
            ce_loss = hidden_states.sum() * 0.0
        return ce_loss, diffusion_loss, n_lat

    # ---------- loss / logging ----------
    def _accumulate(self, prefix: str, ce_loss: torch.Tensor, diffusion_loss: torch.Tensor) -> None:
        # Kept on the GPU; only synced when the Trainer actually logs (no per-micro-step .item()).
        acc = self._loss_acc.get(prefix)
        ce, diff = ce_loss.detach().float(), diffusion_loss.detach().float()
        if acc is None:
            self._loss_acc[prefix] = {"ce": ce.clone(), "diffusion": diff.clone(), "n": 1}
        else:
            acc["ce"] += ce
            acc["diffusion"] += diff
            acc["n"] += 1

    def _pop_accumulated(self, prefix: str) -> Dict[str, float]:
        acc = self._loss_acc.pop(prefix, None)
        if not acc:
            return {}
        vals = torch.stack([acc["ce"], acc["diffusion"]]).cpu() / acc["n"]
        return {f"{prefix}/ce_loss": round(float(vals[0]), 6), f"{prefix}/diffusion_loss": round(float(vals[1]), 6)}

    def compute_loss(self, model, inputs: Dict[str, Any], return_outputs=False, num_items_in_batch: Optional[int] = None):
        ce_loss, diffusion_loss, _ = self.training_forward(model, inputs)
        total = self.args.ce_loss_weight * ce_loss + self.args.diffusion_loss_weight * diffusion_loss
        self._accumulate("train" if model.training else "eval", ce_loss, diffusion_loss)
        return (total, {"ce_loss": ce_loss, "diffusion_loss": diffusion_loss}) if return_outputs else total

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        if "loss" in logs:  # periodic training log
            logs.update(self._pop_accumulated("train"))
        if any(k.startswith("eval_") for k in logs):  # end of an evaluation loop
            logs.update(self._pop_accumulated("eval"))
        super().log(logs, start_time)

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # The stock prediction_step calls model(**inputs), which cannot compute these losses. Loss-only eval.
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad(), self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)
        return loss.detach(), None, None

    def _debug_ce(self, logits: torch.Tensor, labels: torch.Tensor, valid: torch.Tensor) -> None:
        if not self.args.debug_ce_details:
            return
        step = int(self.state.global_step or 0)
        every_n = max(1, int(self.args.debug_ce_every_n_steps or 200))
        if not (step <= 1 or step % every_n == 0):
            return
        with torch.no_grad():
            per_tok = F.cross_entropy(logits, labels, reduction="none")
            k = max(1, int(self.args.debug_ce_topk))
            topk_acc = (logits.topk(k, dim=-1).indices == labels[:, None]).any(-1).float().mean()
            per_row = valid.sum(dim=1)
            max_ex = max(1, int(self.args.debug_ce_max_examples))
            splits = torch.split(per_tok, per_row.tolist())
            per_ex = [round(float(s.mean()), 4) if s.numel() else None for s in splits[:max_ex]]
            logger.info(
                f"CE debug step {step}: tokens_in_loss={labels.numel()}, avg_loss={float(per_tok.mean()):.4f}, "
                f"top{k}_acc={float(topk_acc):.3f}, per_example_avgs={per_ex}"
            )

    # ---------- save / resume ----------
    def _save(self, output_dir: Optional[str] = None, state_dict=None) -> None:
        # Only the fine-tuned parts are written (no full 1.5B copy). Errors propagate: a silent failed save is worse.
        target_dir = output_dir or self.args.output_dir
        os.makedirs(target_dir, exist_ok=True)
        lora_out = save_vibevoice_assets(self.model, target_dir, ema=self.ema, export_ema_head=self.args.export_ema_head)
        torch.save(self.args, os.path.join(target_dir, "training_args.bin"))
        logger.info(f"Saved VibeVoice fine-tune assets to {lora_out}")

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        restored = load_vibevoice_assets(self.model, resume_from_checkpoint, ema=self.ema)
        logger.info(f"Resumed from {resume_from_checkpoint}: restored {restored}")

    def _load_best_model(self):
        best = self.state.best_model_checkpoint
        restored = load_vibevoice_assets(self.model, best, ema=None)
        logger.info(f"Loaded best model from {best}: restored {restored}")


# ================== SETUP HELPERS ==================

def _select_splits(raw, data_args: DataArguments, training_args: CustomTrainingArguments):
    """Accept a Dataset or DatasetDict. Returns (train_ds, eval_ds or None)."""
    if isinstance(raw, DatasetDict):
        if data_args.train_split_name not in raw:
            raise ValueError(f"train split '{data_args.train_split_name}' not in dataset splits {list(raw.keys())}")
        train_ds = raw[data_args.train_split_name]
        eval_ds = raw[data_args.eval_split_name] if (data_args.eval_split_name and data_args.eval_split_name in raw) else None
    elif isinstance(raw, Dataset):
        train_ds, eval_ds = raw, None
    else:
        raise ValueError(f"Unsupported dataset type {type(raw).__name__}; expected datasets.Dataset or DatasetDict")

    wants_eval = bool(training_args.do_eval) or getattr(training_args, "eval_strategy", "no") != "no"
    if wants_eval and eval_ds is None:
        if data_args.eval_split_size and data_args.eval_split_size > 0 and len(train_ds) > 1:
            split = train_ds.train_test_split(test_size=data_args.eval_split_size, seed=training_args.seed)
            train_ds, eval_ds = split["train"], split["test"]
        else:
            raise ValueError("Evaluation requested but there is no eval split; pass --eval_split_size (e.g. 0.02) "
                             f"or provide a '{data_args.eval_split_name}' split.")
    for name, ds in (("train", train_ds), ("eval", eval_ds)):
        if ds is None:
            continue
        for col in (data_args.text_column_name, data_args.audio_column_name):
            if col not in ds.column_names:
                raise ValueError(f"{name} dataset has no column '{col}' (columns: {ds.column_names})")
    return train_ds, eval_ds


def _setup_trainable_params(model: VibeVoiceForConditionalGeneration, model_args: ModelArguments) -> None:
    inner = model.model
    lora_cfg = build_lora_config(model_args)
    tm_lower = [s.strip().lower() for s in model_args.lora_target_modules.split(",") if s.strip()]
    skip_lm_lora = (len(tm_lower) == 0) or all(t in ("none", "off", "disable", "disabled") for t in tm_lower)
    if not skip_lm_lora:
        inner.language_model = get_peft_model(inner.language_model, lora_cfg)
    else:
        logger.info("Skipping LLM LoRA wrapping (lora_target_modules indicates none).")

    # Freeze all, then enable trainable subsets
    for p in model.parameters():
        p.requires_grad = False
    for n, p in inner.language_model.named_parameters():
        if "lora_A" in n or "lora_B" in n:
            p.requires_grad = True

    if model_args.lora_wrap_diffusion_head:
        class _HeadForwardShim(nn.Module):
            def __init__(self, base: nn.Module): super().__init__(); self.base = base
            def forward(self, *args, **kwargs):
                if len(args) >= 3:
                    noisy_images, timesteps, condition = args[:3]
                else:
                    noisy_images = kwargs.get("noisy_images")
                    timesteps = kwargs.get("timesteps")
                    condition = kwargs.get("condition")
                return self.base(noisy_images, timesteps, condition)
        inner.prediction_head = get_peft_model(_HeadForwardShim(inner.prediction_head), build_head_lora_config(model_args))
        for n, p in inner.prediction_head.named_parameters():
            if "lora_A" in n or "lora_B" in n:
                p.requires_grad = True

    if model_args.train_diffusion_head:
        for p in inner.prediction_head.parameters():
            p.requires_grad = True

    if model_args.layers_to_freeze is not None:
        indices_to_freeze = {int(x.strip()) for x in model_args.layers_to_freeze.split(',') if x.strip()}
        head_params = list(inner.prediction_head.named_parameters())
        bad = [i for i in indices_to_freeze if not 0 <= i < len(head_params)]
        if bad:
            raise ValueError(f"--layers_to_freeze indices {bad} out of range (head has {len(head_params)} tensors)")
        for i, (name, param) in enumerate(head_params):
            if i in indices_to_freeze:
                param.requires_grad = False
                logger.info(f"Froze layer [{i}]: {name}")
        logger.info(f"Froze {len(indices_to_freeze)} parameter groups in the diffusion head.")

    for name in ("acoustic_connector", "semantic_connector"):
        for p in getattr(inner, name).parameters():
            p.requires_grad = bool(model_args.train_connectors)

    # Embeddings / LM head and the speech tokenizers always stay frozen
    model.get_input_embeddings().weight.requires_grad_(False)
    if model.get_output_embeddings() is not None:
        model.get_output_embeddings().weight.requires_grad_(False)
    for tok_name, frozen in (("acoustic_tokenizer", model_args.freeze_acoustic_tokenizer),
                             ("semantic_tokenizer", model_args.freeze_semantic_tokenizer)):
        if not frozen:
            logger.warning(f"--freeze_{tok_name} False is ignored: {tok_name} runs under no_grad and is never trained.")
        for p in getattr(inner, tok_name).parameters():
            p.requires_grad = False

    def _count(mod):
        return sum(p.numel() for p in mod.parameters() if p.requires_grad)
    logger.info(f"Trainable by block -> LLM-LoRA: {_count(inner.language_model):,} | diff_head: {_count(inner.prediction_head):,} "
                f"| ac_conn: {_count(inner.acoustic_connector):,} | se_conn: {_count(inner.semantic_connector):,}")
    logger.info("TOTAL trainable: %s", f"{_count(model):,}")


def _log_model_diagnostics(model, tok, run_ce_smoke_test: bool) -> None:
    """Cheap, non-fatal startup diagnostics. The full-model CE smoke test is opt-in (--debug_checks)."""
    in_w, out_w = model.get_input_embeddings().weight, model.get_output_embeddings().weight
    logger.info(f"LM head diagnostics -> shared_params={in_w.data_ptr() == out_w.data_ptr()}, "
                f"tie_word_embeddings={getattr(model.config.decoder_config, 'tie_word_embeddings', None)}")
    vocab_size = int(getattr(model.config.decoder_config, "vocab_size", 0))
    for name in ("speech_start_id", "speech_diffusion_id", "speech_end_id"):
        val = getattr(tok, name, None)
        in_range = isinstance(val, int) and 0 <= val < vocab_size
        logger.info(f"Special token check -> {name}={val}, decoded='{tok.decode([val]) if in_range else None}', in_vocab_range={in_range}")
    if not run_ce_smoke_test:
        return
    with torch.no_grad():
        ids = torch.tensor([tok.encode("The cat sat on the mat.", add_special_tokens=True)], device=model.device)
        out = model.model(inputs_embeds=model.get_input_embeddings()(ids), attention_mask=torch.ones_like(ids), return_dict=True)
        logits = model.lm_head(out.last_hidden_state)
        ce = F.cross_entropy(logits[:, :-1].flatten(0, 1).float(), ids[:, 1:].flatten())
        logger.info(f"Simple text CE loss: {ce.item():.4f}")


def main() -> None:
    parser = HfArgumentParser((ModelArguments, DataArguments, CustomTrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.info("Training/evaluation parameters %s", training_args)
    set_seed(training_args.seed)

    if data_args.voice_prompt_source not in VOICE_PROMPT_SOURCES:
        raise ValueError(f"--voice_prompt_source must be one of {VOICE_PROMPT_SOURCES}")
    if training_args.export_ema_head and not training_args.use_ema:
        raise ValueError("--export_ema_head requires --use_ema")
    if training_args.export_ema_head and model_args.lora_wrap_diffusion_head:
        raise ValueError("--export_ema_head is only supported with --train_diffusion_head (full head), not a LoRA head")

    # Configure gradient clipping
    if not training_args.gradient_clipping:
        training_args.max_grad_norm = 0.0
        logger.info("Gradient clipping disabled (set max_grad_norm=0.0). Use --gradient_clipping to enable.")
    else:
        if training_args.max_grad_norm is None or training_args.max_grad_norm <= 0:
            training_args.max_grad_norm = 1.0
        logger.info(f"Gradient clipping enabled: max_grad_norm={training_args.max_grad_norm}")

    # Load processor
    processor_path = model_args.processor_name_or_path or model_args.model_name_or_path
    if processor_path is None:
        raise ValueError("--model_name_or_path (or --processor_name_or_path) must be provided")
    processor: VibeVoiceProcessor = VibeVoiceProcessor.from_pretrained(processor_path)

    tok = processor.tokenizer
    for required in ["speech_start_id", "speech_diffusion_id", "speech_end_id"]:
        if getattr(tok, required, None) is None:
            raise RuntimeError(f"Tokenizer missing required special id: {required}")

    # Load model
    if model_args.model_name_or_path is None:
        raise ValueError("--model_name_or_path is required to load VibeVoice base model")
    dtype = torch.bfloat16 if training_args.bf16 else (torch.float16 if training_args.fp16 else torch.float32)
    model = VibeVoiceForConditionalGeneration.from_pretrained(
        model_args.model_name_or_path,
        torch_dtype=dtype,
        attn_implementation=model_args.attn_implementation,
    )
    logger.info(f"LM attention implementation: {model.model.language_model.config._attn_implementation}")
    _patch_acoustic_encode_for_legacy_indexing(model, logger)

    # Hard-tie LM head to input embeddings (the checkpoint has no separate lm_head weight)
    emb_module, head_module = model.get_input_embeddings(), model.get_output_embeddings()
    if emb_module.weight.shape == head_module.weight.shape and emb_module.weight.data_ptr() != head_module.weight.data_ptr():
        head_module.weight = emb_module.weight
        logger.info("Force-tied LM head weight to input embeddings (pointer share).")

    _log_model_diagnostics(model, tok, run_ce_smoke_test=training_args.debug_checks)

    if training_args.do_train:
        model.config.use_cache = False
        model.model.language_model.config.use_cache = False

    _setup_trainable_params(model, model_args)

    # Gradient checkpointing must be configured BEFORE the Trainer is built. Non-reentrant checkpointing works with
    # frozen inputs (LoRA), and input grads make sure the checkpointed blocks still receive gradients.
    if training_args.gradient_checkpointing:
        gc_kwargs = training_args.gradient_checkpointing_kwargs or {"use_reentrant": False}
        training_args.gradient_checkpointing_kwargs = gc_kwargs
        model.model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=gc_kwargs)
        if gc_kwargs.get("use_reentrant", True):
            # reentrant checkpointing needs a grad-requiring input, otherwise frozen blocks get no gradient
            model.model.language_model.enable_input_require_grads()
        # Trainer would otherwise call gradient_checkpointing_enable() on the wrapper model again.
        training_args.gradient_checkpointing = False
        logger.info(f"Gradient checkpointing enabled on the language model with {gc_kwargs}")

    # Datasets
    verification_mode = VerificationMode.NO_CHECKS if data_args.ignore_verifications else VerificationMode.BASIC_CHECKS
    if data_args.train_jsonl is not None:
        data_files: Dict[str, str] = {data_args.train_split_name: data_args.train_jsonl}
        if data_args.validation_jsonl is not None and data_args.eval_split_name:
            data_files[data_args.eval_split_name] = data_args.validation_jsonl
        raw = load_dataset("json", data_files=data_files, verification_mode=verification_mode, cache_dir=model_args.cache_dir)
    elif data_args.dataset_name is None:
        raise ValueError("Provide --dataset_name (a save_to_disk directory) or --train_jsonl/--validation_jsonl.")
    elif os.path.isdir(data_args.dataset_name):
        raw = load_from_disk(data_args.dataset_name)
    else:
        raw = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, verification_mode=verification_mode,
            cache_dir=model_args.cache_dir, token=os.environ.get("HF_TOKEN"),
        )
    train_ds, eval_ds = _select_splits(raw, data_args, training_args)
    logger.info(f"Train examples: {len(train_ds)} | eval examples: {len(eval_ds) if eval_ds is not None else 0} "
                f"| voice_prompt_source={data_args.voice_prompt_source}")

    ds_kwargs = dict(
        text_column=data_args.text_column_name,
        audio_column=data_args.audio_column_name,
        voice_prompts_column=data_args.voice_prompts_column_name,
        voice_prompt_source=data_args.voice_prompt_source,
        prompt_min_sec=data_args.voice_prompt_min_sec,
        prompt_max_sec=data_args.voice_prompt_max_sec,
    )
    train_dataset = VibeVoiceDataset(train_ds, **ds_kwargs)
    eval_dataset = VibeVoiceDataset(eval_ds, **ds_kwargs) if eval_ds is not None else None

    data_collator = VibeVoiceCollator(
        processor=processor,
        max_length=data_args.max_length,
        speech_compress_ratio=getattr(processor, "speech_tok_compress_ratio", 3200),
        semantic_vae_dim=int(getattr(model.config, "semantic_vae_dim", 128)),
        debug_checks=training_args.debug_checks,
        voice_prompt_drop_rate=data_args.voice_prompt_drop_rate,
    )

    ce_keep_ids = None
    if training_args.ce_on_speech_control_only:
        eos = getattr(tok, "eos_id", None)
        eos = eos if eos is not None else tok.eos_token_id
        ce_keep_ids = sorted({int(tok.speech_end_id), int(eos)})
        logger.info(f"CE restricted to speech control tokens {ce_keep_ids} (+ target speech placeholders)")

    ema = HeadEMA(decay=training_args.ema_decay) if training_args.use_ema else None
    callbacks: List[TrainerCallback] = [LoRADebugCallback(log_every_n_steps=int(training_args.logging_steps or 50))]
    if ema is not None:
        callbacks.insert(0, EmaCallback(ema))

    trainer = VibeVoiceTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=callbacks,
        ema=ema,
        ce_keep_token_ids=ce_keep_ids,
        cache_target_features=training_args.cache_target_features,
    )

    if training_args.target_feature_cache_dir and training_args.do_train:
        setup_target_feature_cache(trainer, model, train_dataset, train_ds, data_args, training_args)

    if training_args.debug_save:
        debug_dir = os.path.join(training_args.output_dir, "debug_initial")
        logger.info(f"[debug_save] Saving initial (pre-training) model components to: {debug_dir}")
        save_vibevoice_assets(model, debug_dir, ema=None)

    if training_args.do_train:
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
        final_dir = save_vibevoice_assets(model, training_args.output_dir, ema=ema, export_ema_head=training_args.export_ema_head)
        trainer.save_state()
        logger.info(f"Final assets saved to {final_dir} (exported head: {'EMA' if training_args.export_ema_head else 'trained'})")

    if training_args.do_eval and eval_dataset is not None:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)


if __name__ == "__main__":
    main()
