from datasets import load_dataset, Features, Value, Audio, Sequence, Dataset, load_from_disk, concatenate_datasets
import os
import soundfile as sf

CHUNK_SIZE = 1000
TOTAL_RECORDS = 20000
SAVE_PATH = "./vibevoice_dataset/alvanlii_cantonese_radio/checkpoint"

print("Loading raw dataset...")
features = Features({
    "source": Value("string"),
    "filename": Value("string"),
    "order_index": Value("string"),
    "link": Value("string"),
    "transcript_whisper": Value("string"),
    "audio": Audio(sampling_rate=16000),
    "c50": Value("float32"),            
    "snr": Value("float32"),            
    "speech_duration": Value("float32"),
    "emotion_emotion2vec": Value("string"),
    "transcript_sensevoice": Value("string"),
    "emotion_sensevoice": Sequence(Value("string")),
    "event_sensevoice": Sequence(Value("string"))
})

def get_already_processed_count():
    chunks = [d for d in os.listdir(SAVE_PATH) if d.startswith("chunk_")]
    return len(chunks) * CHUNK_SIZE

streamed_ds = load_dataset("alvanlii/cantonese-radio", split="train", features=features, streaming=True, token="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")

def format_for_vibe(example):
    return {
        "text": f"Speaker 1: {example['transcript_sensevoice']}",
        "audio": example["audio"]
    }
    

formatted_stream = streamed_ds.filter(lambda x: x['c50'] >= 55 and 'music' not in x['event_sensevoice'])
formatted_stream = formatted_stream.map(format_for_vibe)

start_idx = get_already_processed_count()
current_stream = formatted_stream.skip(start_idx)

for i in range(start_idx, TOTAL_RECORDS, CHUNK_SIZE):
    print(f"Processing records {i} to {i + CHUNK_SIZE}...")
    chunk_data = list(current_stream.take(CHUNK_SIZE))
    chunk_ds = Dataset.from_list(chunk_data)
    chunk_ds.save_to_disk(f"{SAVE_PATH}/chunk_{i}")
    print(f"Chunk {i} saved!")

print("All chunks finished!")

all_chunks = []
for folder in sorted(os.listdir(SAVE_PATH)):
    all_chunks.append(load_from_disk(f"{SAVE_PATH}/{folder}"))

final_dataset = concatenate_datasets(all_chunks)
final_dataset.save_to_disk("./vibevoice_dataset/alvanlii_cantonese_radio")