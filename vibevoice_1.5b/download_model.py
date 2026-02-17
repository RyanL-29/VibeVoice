import os
import json
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
import torch

REPO_ID = "microsoft/VibeVoice-1.5B"
LOCAL_DIR = "./vibevoice_1.5b"
OUTPUT_FILE = "./vibevoice_1.5b/vibevoice_1.5b_merged.safetensors"

def download_and_merge():
    print(f"Downloading shards from {REPO_ID}...")
    snapshot_download(repo_id=REPO_ID, local_dir=LOCAL_DIR, allow_patterns=["*.safetensors", "*.json"])

    index_path = os.path.join(LOCAL_DIR, "model.safetensors.index.json")
    with open(index_path, "r") as f:
        index_data = json.load(f)

    weight_map = index_data["weight_map"]
    shards = sorted(list(set(weight_map.values())))
    
    merged_weights = {}

    # 3. Load each shard and add to the master dictionary
    print(f"Found {len(shards)} shards. Merging...")
    for shard_name in shards:
        shard_path = os.path.join(LOCAL_DIR, shard_name)
        print(f"Loading {shard_name}...")
        shard_tensors = load_file(shard_path)
        merged_weights.update(shard_tensors)

    # 4. Save the single merged file
    print(f"Saving merged model to {OUTPUT_FILE}...")
    save_file(merged_weights, OUTPUT_FILE)
    print("Success! You now have a single .safetensors file.")

if __name__ == "__main__":
    download_and_merge()