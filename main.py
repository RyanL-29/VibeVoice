import copy
import torch
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger(__name__)
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference

def main():
    processor = VibeVoiceProcessor.from_pretrained("vibevoice_1.5b")
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        "vibevoice_1.5b",
        torch_dtype=torch.float32,
        device_map="cpu",
        attn_implementation="sdpa",
    )

    model.eval()
    model.set_ddpm_inference_steps(num_steps=5)

    if hasattr(model.model, 'language_model'):
        print(f"Language model attention: {model.model.language_model.config._attn_implementation}")
        
    voice_samples = ["vibevoice_realtime/voices/testing.wav"]
    print(f"Start generate")
    inputs = processor(
        text=["Speaker 1: Manufactured by JRC (New Japan Radio), it uses oxygen-free copper lead frames and is tuned specifically for audio appreciation rather than just raw data specs."],
        # cached_prompt=all_prefilled_outputs,
        # voice_samples=[voice_samples],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to("cpu")
    outputs = model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=1.3,
        tokenizer=processor.tokenizer,
        generation_config={'do_sample': False},
        verbose=True,
        is_prefill=True
        # all_prefilled_outputs=copy.deepcopy(all_prefilled_outputs) if all_prefilled_outputs is not None else None,
    )
    processor.save_audio(
        outputs.speech_outputs[0], # First (and only) batch item
        output_path="testing.wav",
    )
    
if __name__ == "__main__":
    main()