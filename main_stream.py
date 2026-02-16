import copy
import torch
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger(__name__)
from vibevoice.modular.modeling_vibevoice_streaming_inference import VibeVoiceStreamingForConditionalGenerationInference
from vibevoice.processor.vibevoice_streaming_processor import VibeVoiceStreamingProcessor
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
from vibevoice.modular.modeling_vibevoice import VibeVoiceForConditionalGeneration

def main():
    processor = VibeVoiceStreamingProcessor.from_pretrained("vibevoice_realtime")
    model = VibeVoiceStreamingForConditionalGenerationInference.from_pretrained(
        "vibevoice_realtime",
        torch_dtype=torch.float32,
        device_map="cpu",
        attn_implementation="sdpa",
    )

    model.eval()
    model.set_ddpm_inference_steps(num_steps=5)

    if hasattr(model.model, 'language_model'):
        print(f"Language model attention: {model.model.language_model.config._attn_implementation}")
        
    voice_sample = "vibevoice_realtime/voices/en-Frank_man.pt"
    print(f"Using voice preset for Carter: {voice_sample}")
    all_prefilled_outputs = torch.load(voice_sample, map_location="cpu", weights_only=False)
    print(f"Start generate")
    inputs = processor.process_input_with_cached_prompt(
        text="這表示你除了去百貨公司和公園，可能還去了其他地方，只是舉這兩個為例。".strip().replace("’", "'").replace('“', '"').replace('”', '"'),
        cached_prompt=all_prefilled_outputs,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    for k, v in inputs.items():
        if torch.is_tensor(v):
            inputs[k] = v.to("cpu")
    
    print(f"Starting generation with cfg_scale: 1.5")
    outputs = model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=1.5,
        tokenizer=processor.tokenizer,
        generation_config={'do_sample': False},
        verbose=True,
        all_prefilled_outputs=copy.deepcopy(all_prefilled_outputs) if all_prefilled_outputs is not None else None,
    )
    processor.save_audio(
        outputs.speech_outputs[0], # First (and only) batch item
        output_path="testing.wav",
    )
    
if __name__ == "__main__":
    main()