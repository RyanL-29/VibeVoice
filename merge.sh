cd community && \
uv run vibevoice/scripts/merge_vibevoice_models.py \
     --base_model_path /home/administrator/VibeVoice/vibevoice_1.5b \
     --checkpoint_path /home/administrator/VibeVoice/vibevoice_1.5b_cantonese_train/lora \
     --output_path /home/administrator/VibeVoice/vibevoice_1.5b_cantonese_train/merge