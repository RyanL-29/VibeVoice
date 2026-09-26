import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import warnings
import random

try:
    import librosa  # type: ignore
except Exception:  # pragma: no cover
    librosa = None  # Fallback: user must install librosa when using local audio paths

try:
    import resampy  # type: ignore
except Exception:  # pragma: no cover
    resampy = None


def _resample_if_needed(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wav.astype(np.float32, copy=False)
    if resampy is not None:
        return resampy.resample(wav.astype(np.float32), orig_sr, target_sr)
    if librosa is not None:
        return librosa.resample(y=wav.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr)
    warnings.warn(
        "No resampler available; treating audio as target_sr without resampling. Install resampy or librosa.",
        RuntimeWarning,
    )
    return wav.astype(np.float32, copy=False)


VOICE_PROMPT_SOURCES = ("column", "self_crop", "other")

# Identifies how prepare_target_waveform() builds the target audio; part of the precomputed-feature cache key, so
# change it whenever that function (or the silence/crossfade defaults below) changes.
TARGET_WAVEFORM_SPEC = "24k-float32+silence(pre=0.25,xfade=0.25/0.25,post=0.75)"


def prepare_target_waveform(audio: Union[np.ndarray, torch.Tensor, str, Dict[str, Any]]) -> np.ndarray:
    """The exact waveform the collator encodes for a target clip (24 kHz float32 + leading/trailing silence).

    Shared by the collator and the offline feature precompute, so precomputed encoder features always match."""
    return _load_audio_to_24k(audio, target_sr=24000, augment_with_silence=True)


def _self_crop_prompt(wav_array: np.ndarray, target_sr: int = 24000) -> Optional[np.ndarray]:
    """Legacy prompt: a random crop (1/4..1/2 of the clip, capped at 5..15 s) of the *target* clip itself."""
    audio_len_seconds = len(wav_array) / target_sr
    min_len_sec = min(5.0, audio_len_seconds / 4.0)
    max_len_sec = min(15.0, audio_len_seconds / 2.0)
    if min_len_sec > max_len_sec:
        min_len_sec = max_len_sec
    max_len_sec = min(max_len_sec, audio_len_seconds)
    if max_len_sec <= 0.1:
        return None
    prompt_len_samples = int(random.uniform(min_len_sec, max_len_sec) * target_sr)
    start_sample = random.randint(0, len(wav_array) - prompt_len_samples)
    return wav_array[start_sample : start_sample + prompt_len_samples]


def _random_crop(
    wav_array: np.ndarray,
    *,
    target_sr: int = 24000,
    min_sec: float = 3.0,
    max_sec: float = 10.0,
) -> Optional[np.ndarray]:
    """Random crop of length U(min_sec, max_sec) seconds (clamped to the clip length)."""
    total = len(wav_array)
    if total < int(0.1 * target_sr):
        return None
    hi = min(int(max_sec * target_sr), total)
    lo = min(int(min_sec * target_sr), hi)
    crop_len = random.randint(lo, hi)
    start = random.randint(0, total - crop_len)
    return wav_array[start : start + crop_len]


# Lightweight HF-style dataset wrapper (optional). Trainer can also pass raw HF datasets directly.
class VibeVoiceDataset:
    """Wraps an HF dataset and yields {text, audio (np.float32 @ 24 kHz), voice_prompts}.

    voice_prompt_source:
      - "column":    use ``voice_prompts_column`` if present/non-empty, else fall back to "self_crop".
      - "self_crop": crop of the target clip itself (legacy behaviour; the prompt overlaps the target).
      - "other":     random crop (up to ``prompt_max_sec``) of a *different*, randomly chosen clip.
                     Assumes a single-speaker dataset (any other clip is the same voice).
    Audio is decoded/resampled exactly once per item here, so the collator only pads.
    """

    def __init__(
        self,
        dataset: Any,
        text_column: str = "text",
        audio_column: str = "audio",
        voice_prompts_column: Optional[str] = "voice_prompts",
        voice_prompt_source: str = "column",
        prompt_min_sec: float = 3.0,
        prompt_max_sec: float = 10.0,
        target_sr: int = 24000,
    ) -> None:
        if voice_prompt_source not in VOICE_PROMPT_SOURCES:
            raise ValueError(f"voice_prompt_source must be one of {VOICE_PROMPT_SOURCES}, got {voice_prompt_source!r}")
        self.dataset = dataset
        self.text_column = text_column
        self.audio_column = audio_column
        self.voice_prompts_column = voice_prompts_column
        self.voice_prompt_source = voice_prompt_source
        self.prompt_min_sec = float(prompt_min_sec)
        self.prompt_max_sec = float(prompt_max_sec)
        self.target_sr = int(target_sr)

    def __len__(self) -> int:
        return len(self.dataset)

    def target_waveform(self, idx: int) -> np.ndarray:
        """Target clip `idx` exactly as the collator will encode it (used by the feature precompute)."""
        wav = _load_audio_to_24k(self.dataset[idx][self.audio_column], target_sr=self.target_sr)
        return prepare_target_waveform(wav)

    def _other_clip_prompt(self, idx: int) -> Optional[np.ndarray]:
        n = len(self.dataset)
        if n < 2:
            return None
        j = random.randrange(n - 1)
        if j >= idx:
            j += 1  # uniform over all indices except idx
        other_wav = _load_audio_to_24k(self.dataset[j][self.audio_column], target_sr=self.target_sr)
        return _random_crop(
            other_wav, target_sr=self.target_sr, min_sec=self.prompt_min_sec, max_sec=self.prompt_max_sec
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.dataset[idx]
        data: Dict[str, Any] = {}
        data["text"] = item[self.text_column]
        data["index"] = int(idx)  # lets the trainer cache the (deterministic) encoder features of the target clip
        wav_array = _load_audio_to_24k(item[self.audio_column], target_sr=self.target_sr)
        data["audio"] = wav_array

        try:
            if self.voice_prompt_source == "other":
                prompt = self._other_clip_prompt(idx)
                data["voice_prompts"] = [prompt] if prompt is not None else None
                return data

            user_provided_prompt = None
            if self.voice_prompt_source == "column" and self.voice_prompts_column and self.voice_prompts_column in item:
                user_provided_prompt = item[self.voice_prompts_column]
            if user_provided_prompt:
                data["voice_prompts"] = (
                    user_provided_prompt if isinstance(user_provided_prompt, list) else [user_provided_prompt]
                )
            else:
                # "self_crop", or "column" with no prompt available for this row.
                prompt = _self_crop_prompt(wav_array, target_sr=self.target_sr)
                data["voice_prompts"] = [prompt] if prompt is not None else None
        except Exception as e:
            warnings.warn(f"Could not create voice prompt for item {idx}: {e}")
            data["voice_prompts"] = None
        return data



def _apply_silence_with_crossfade(
    wav: np.ndarray,
    *,
    sample_rate: int,
    pre_silence_sec: float = 0.25,
    pre_crossfade_sec: float = 0.25,
    post_crossfade_sec: float = 0.25,
    post_silence_sec: float = 0.75,
) -> np.ndarray:
    """Pad audio with leading/trailing silence and apply crossfades.

    Structure: [pre_silence][pre_crossfade][audio_body][post_crossfade][post_silence]
    Crossfades blend the audio with silence linearly to avoid hard edges.
    """

    wav = np.asarray(wav, dtype=np.float32).reshape(-1)

    start_sil_samples = int(round(pre_silence_sec * sample_rate))
    end_sil_samples = int(round(post_silence_sec * sample_rate))
    pre_crossfade_samples = int(round(pre_crossfade_sec * sample_rate))
    post_crossfade_samples = int(round(post_crossfade_sec * sample_rate))

    total_len = wav.shape[0]
    if total_len == 0:
        pieces: List[np.ndarray] = []
        if start_sil_samples > 0:
            pieces.append(np.zeros(start_sil_samples, dtype=np.float32))
        if end_sil_samples > 0:
            pieces.append(np.zeros(end_sil_samples, dtype=np.float32))
        return np.concatenate(pieces) if pieces else wav

    start_len = min(pre_crossfade_samples, total_len)
    remaining_after_start = max(total_len - start_len, 0)
    end_len = min(post_crossfade_samples, remaining_after_start)
    middle_end_idx = total_len - end_len

    start_segment = wav[:start_len]
    middle_segment = wav[start_len:middle_end_idx]
    end_segment = wav[middle_end_idx:]

    def _linear_fade(num_samples: int, start: float, end: float) -> np.ndarray:
        if num_samples <= 0:
            return np.zeros((0,), dtype=np.float32)
        return np.linspace(start, end, num_samples, endpoint=True, dtype=np.float32)

    start_crossfade = start_segment * _linear_fade(start_len, 0.0, 1.0)
    end_crossfade = end_segment * _linear_fade(end_segment.shape[0], 1.0, 0.0)

    pieces: List[np.ndarray] = []
    if start_sil_samples > 0:
        pieces.append(np.zeros(start_sil_samples, dtype=np.float32))
    if start_crossfade.size > 0:
        pieces.append(start_crossfade.astype(np.float32, copy=False))
    if middle_segment.size > 0:
        pieces.append(middle_segment.astype(np.float32, copy=False))
    if end_crossfade.size > 0:
        pieces.append(end_crossfade.astype(np.float32, copy=False))
    if end_sil_samples > 0:
        pieces.append(np.zeros(end_sil_samples, dtype=np.float32))

    return np.concatenate(pieces)


def _load_audio_to_24k(
    audio: Union[str, np.ndarray, torch.Tensor, Dict[str, Any]],
    *,
    target_sr: int = 24000,
    augment_with_silence: bool = False,
) -> np.ndarray:
    if isinstance(audio, np.ndarray):
        wav_out = audio.astype(np.float32)
    elif isinstance(audio, torch.Tensor):
        wav_out = audio.detach().cpu().float().numpy()
    elif isinstance(audio, str):
        if librosa is None:
            raise RuntimeError("librosa is required to load audio file paths. Please pip install librosa.")
        wav, sr = librosa.load(audio, sr=None, mono=True)
        wav_out = _resample_if_needed(wav, int(sr), target_sr)
    elif isinstance(audio, dict) and "array" in audio and "sampling_rate" in audio:
        arr = np.asarray(audio["array"], dtype=np.float32)
        sr = int(audio["sampling_rate"])
        wav_out = _resample_if_needed(arr, sr, target_sr)
    else:
        raise ValueError(f"Unsupported audio type: {type(audio)}")

    wav_out = np.asarray(wav_out, dtype=np.float32)

    if augment_with_silence:
        wav_out = _apply_silence_with_crossfade(wav_out, sample_rate=target_sr)

    return wav_out


@dataclass
class VibeVoiceCollator:
    processor: Any  # VibeVoiceProcessor
    max_length: Optional[int] = None
    speech_compress_ratio: int = 3200
    semantic_vae_dim: int = 128
    compute_semantics: bool = False
    debug_checks: bool = False

    text_field: str = "text"
    audio_field: str = "audio"
    voice_prompts_field: str = "voice_prompts"
    voice_prompt_drop_rate: float = 0.0

    def __call__(self, features: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        batch_size = len(features)

        sample_input_ids: List[List[int]] = []
        sample_attention_masks: List[List[int]] = []
        sample_acoustic_input_masks: List[List[bool]] = []
        sample_acoustic_loss_masks: List[List[bool]] = []

        all_speech_waveforms: List[np.ndarray] = []
        all_speech_latent_lengths: List[int] = []
        per_segment_is_target: List[bool] = []

        for ex in features:
            text: str = ex.get(self.text_field, "")
            voice_prompts: Optional[List[Union[str, np.ndarray, torch.Tensor]]] = ex.get(self.voice_prompts_field)
            target_audio: Union[str, np.ndarray, torch.Tensor, Dict[str, Any]] = ex.get(self.audio_field)

            # Clamp drop rate for safety
            _drop_rate = self.voice_prompt_drop_rate
            if _drop_rate < 0.0:
                _drop_rate = 0.0
            elif _drop_rate > 1.0:
                _drop_rate = 1.0

            proc = self.processor(
                text=[text],
                voice_samples=[voice_prompts] if voice_prompts is not None and random.random() >= _drop_rate else None,
                padding=False,
                truncation=False,
                max_length=self.max_length,
                return_tensors="pt",
            )

            ids = proc["input_ids"][0].tolist()
            attn = proc.get("attention_mask", torch.ones_like(proc["input_ids"]))[0].tolist()
            speech_input_mask = proc.get("speech_input_mask")
            if speech_input_mask is None:
                speech_input_mask = torch.zeros_like(proc["input_ids"], dtype=torch.bool)
            speech_input_mask_list = speech_input_mask[0].tolist()

            wav_target = prepare_target_waveform(target_audio)
            # The acoustic/semantic tokenizers are causal convs with stride = speech_compress_ratio, so a clip of
            # N samples yields exactly ceil(N / ratio) latent frames (same rule the processor uses for prompts).
            target_latent_len = max(1, int(math.ceil(len(wav_target) / float(self.speech_compress_ratio))))

            speech_diff_id = self.processor.tokenizer.speech_diffusion_id
            target_placeholders = [speech_diff_id] * target_latent_len

            ids_extended = ids + target_placeholders
            attn_extended = attn + [1] * target_latent_len

            acoustic_input_mask = speech_input_mask_list + [True] * target_latent_len
            acoustic_loss_mask = ([False] * len(speech_input_mask_list)) + [True] * target_latent_len

            speech_end_id = self.processor.tokenizer.speech_end_id
            ids_extended.append(speech_end_id)
            attn_extended.append(1)
            acoustic_input_mask.append(False)
            acoustic_loss_mask.append(False)

            # Ensure text decoding sees an explicit end-of-sequence token after speech output.
            eos_token_id = getattr(self.processor.tokenizer, "eos_id", None)
            if eos_token_id is None:
                eos_token_id = getattr(self.processor.tokenizer, "eos_token_id", None)
            if eos_token_id is not None and eos_token_id >= 0:
                ids_extended.append(eos_token_id)
                attn_extended.append(1)
                acoustic_input_mask.append(False)
                acoustic_loss_mask.append(False)

            if self.max_length is not None and len(ids_extended) > self.max_length:
                cut = len(ids_extended) - int(self.max_length)
                leading_non_acoustic = 0
                for v in acoustic_input_mask:
                    if v:
                        break
                    leading_non_acoustic += 1
                if cut > leading_non_acoustic:
                    raise ValueError(
                        f"--max_length={self.max_length} would truncate into acoustic tokens. "
                        f"Needed cut={cut}, but only {leading_non_acoustic} leading non-acoustic tokens available. "
                        "Increase max_length or shorten text/voice-prompt preamble."
                    )
                ids_extended = ids_extended[cut:]
                attn_extended = attn_extended[cut:]
                acoustic_input_mask = acoustic_input_mask[cut:]
                acoustic_loss_mask = acoustic_loss_mask[cut:]

            sample_input_ids.append(ids_extended)
            sample_attention_masks.append(attn_extended)
            sample_acoustic_input_masks.append(acoustic_input_mask)
            sample_acoustic_loss_masks.append(acoustic_loss_mask)

            voice_speeches = []
            voice_latent_lengths = []
            if proc.get("speech_tensors") is not None:
                voice_np = proc["speech_tensors"].cpu().numpy()
                voice_masks = proc["speech_masks"].cpu().numpy().astype(bool)
                for seg_idx in range(voice_np.shape[0]):
                    voice_speeches.append(voice_np[seg_idx])
                    voice_latent_lengths.append(int(voice_masks[seg_idx].sum()))

            all_speech_waveforms.extend(voice_speeches)
            all_speech_latent_lengths.extend(voice_latent_lengths)
            per_segment_is_target.extend([False] * len(voice_speeches))

            all_speech_waveforms.append(wav_target)
            all_speech_latent_lengths.append(target_latent_len)
            per_segment_is_target.append(True)

        max_seq_len = max(len(x) for x in sample_input_ids)
        padded_input_ids = []
        padded_attention_masks = []
        padded_acoustic_input_masks = []
        padded_acoustic_loss_masks = []
        tok = self.processor.tokenizer
        pad_token_id = getattr(tok, "pad_token_id", None)
        if pad_token_id is None or pad_token_id < 0:
            pad_token_id = getattr(tok, "eos_token_id", None)
            if pad_token_id is None or pad_token_id < 0:
                raise ValueError(
                    "Tokenizer has no pad_token_id or eos_token_id; please set one or pass a valid pad id."
                )
        for ids, attn, ain_mask, aloss_mask in zip(
            sample_input_ids, sample_attention_masks, sample_acoustic_input_masks, sample_acoustic_loss_masks
        ):
            pad_len = max_seq_len - len(ids)
            padded_input_ids.append(ids + [pad_token_id] * pad_len)
            padded_attention_masks.append(attn + [0] * pad_len)
            padded_acoustic_input_masks.append(ain_mask + [False] * pad_len)
            padded_acoustic_loss_masks.append(aloss_mask + [False] * pad_len)

        input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long)
        attention_mask_tensor = torch.tensor(padded_attention_masks, dtype=torch.long)
        acoustic_input_mask_tensor = torch.tensor(padded_acoustic_input_masks, dtype=torch.bool)
        acoustic_loss_mask_tensor = torch.tensor(padded_acoustic_loss_masks, dtype=torch.bool)

        if all_speech_waveforms:
            max_wave_len = max(w.shape[0] for w in all_speech_waveforms)
            padded_speeches = np.zeros((len(all_speech_waveforms), max_wave_len), dtype=np.float32)
            for i, w in enumerate(all_speech_waveforms):
                L = w.shape[0]
                padded_speeches[i, :L] = w

            max_latent_len = max(all_speech_latent_lengths) if all_speech_latent_lengths else 1
            speech_masks_np = np.zeros((len(all_speech_waveforms), max_latent_len), dtype=np.bool_)
            for i, L_lat in enumerate(all_speech_latent_lengths):
                speech_masks_np[i, :L_lat] = True

            speech_tensors_tensor = torch.from_numpy(padded_speeches)
            # Exact per-segment sample counts (plain ints, stay on the host): the trainer encodes every segment at its
            # true length, which is faster than encoding the zero-padded batch and matches inference (no padding
            # leaking into the last latent frame).
            speech_lengths = [int(w.shape[0]) for w in all_speech_waveforms]
            target_segment_rows = [i for i, is_t in enumerate(per_segment_is_target) if is_t]
            # one target segment per example, in order -> dataset index of each target (None if unknown)
            target_keys = [ex.get("index") for ex in features]
            speech_masks_tensor = torch.tensor(speech_masks_np, dtype=torch.bool)

            speeches_loss_input_np = np.zeros_like(speech_masks_np, dtype=np.bool_)
            for i, is_target in enumerate(per_segment_is_target):
                if is_target:
                    speeches_loss_input_np[i] = speech_masks_np[i]
            speeches_loss_input_tensor = torch.tensor(speeches_loss_input_np, dtype=torch.bool)
        else:
            speech_tensors_tensor = None
            speech_masks_tensor = None
            speeches_loss_input_tensor = None
            speech_lengths = None
            target_segment_rows = None
            target_keys = None

        # Semantic features are NOT computed here: the semantic tokenizer is a GPU model and is run batched inside
        # the trainer's forward (matching inference: target frames get acoustic+semantic, prompt frames acoustic only).
        # `compute_semantics` / `semantic_vae_dim` are kept only for backward compatibility of the constructor.
        speech_semantic_tensors = None

        if self.debug_checks:
            if not bool((input_ids_tensor >= 0).all()):
                raise ValueError("input_ids contains negative indices")
            if speech_tensors_tensor is not None and speech_tensors_tensor.dim() != 2:
                raise ValueError("Expected speech_tensors 2D [segments, samples]")
            n_placeholders = int(acoustic_input_mask_tensor.sum())
            n_latents = int(speech_masks_tensor.sum()) if speech_masks_tensor is not None else 0
            if n_placeholders != n_latents:
                raise ValueError(f"acoustic_input_mask has {n_placeholders} placeholders but speech_masks has {n_latents} latents")

        return {
            "input_ids": input_ids_tensor,
            "attention_mask": attention_mask_tensor,
            "speech_tensors": speech_tensors_tensor,
            "speech_masks": speech_masks_tensor,
            "speech_lengths": speech_lengths,
            "target_segment_rows": target_segment_rows,
            "target_keys": target_keys,
            "speech_semantic_tensors": speech_semantic_tensors,
            "acoustic_input_mask": acoustic_input_mask_tensor,
            "acoustic_loss_mask": acoustic_loss_mask_tensor,
            "speeches_loss_input": speeches_loss_input_tensor,
        }