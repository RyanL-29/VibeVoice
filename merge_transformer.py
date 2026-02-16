import torch
from safetensors.torch import load_file, save_file
import os

# Update these filenames to match yours exactly
files = [
    "vibevoice_1.5b/model-00001-of-00003.safetensors",
    "vibevoice_1.5b/model-00002-of-00003.safetensors",
    "vibevoice_1.5b/model-00003-of-00003.safetensors"
]

merged_dict = {}

print("Merging shards...")
for f in files:
    if os.path.exists(f):
        shard = load_file(f)
        merged_dict.update(shard)
        print(f"Loaded {f}")
    else:
        print(f"Error: {f} not found!")

# Save as the single file the error is looking for
save_file(merged_dict, "vibevoice_1.5b/model.safetensors")
print("Success! Created model.safetensors")