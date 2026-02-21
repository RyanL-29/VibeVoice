cd community && \
uv run vibevoice/scripts/merge_vibevoice_models.py \
     --base_model_path /root/VibeVoice/vibevoice_1.5b \
     --checkpoint_path /root/VibeVoice/vibevoice_1.5b_cantonese_train/checkpoint-500/lora \
     --output_path /root/VibeVoice/vibevoice_1.5b_cantonese_train/merge