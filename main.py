import copy
import torch
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger(__name__)
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference

def main():
    processor = VibeVoiceProcessor.from_pretrained("vibevoice_1.5b_cantonese_train/merge")
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        "vibevoice_1.5b_cantonese_train/merge",
        # "vibevoice_1.5b",
        torch_dtype=torch.float32,
        device_map="cuda",
        attn_implementation="sdpa",
    )

    model.eval()
    model.set_ddpm_inference_steps(num_steps=5)

    if hasattr(model.model, 'language_model'):
        print(f"Language model attention: {model.model.language_model.config._attn_implementation}")
        
    voice_samples = ["vibevoice_realtime/voices/sample.wav"]
    print(f"Start generate")
    inputs = processor(
        text=["Speaker 1: 警告!機房溫度過高!目前讀數攝氏四十二度"],
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to("cuda")
    outputs = model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=1.5,
        tokenizer=processor.tokenizer,
        generation_config={'do_sample': False},
        verbose=True,
        is_prefill=True
    )
    processor.save_audio(
        outputs.speech_outputs[0], # First (and only) batch item
        output_path="testing.wav",
    )
    
if __name__ == "__main__":
    main()