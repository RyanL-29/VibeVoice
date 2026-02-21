from datasets import load_dataset, Features, Value, Audio, Sequence, Dataset, load_from_disk, concatenate_datasets
import os
import soundfile as sf

CHUNK_SIZE = 1000
TOTAL_RECORDS = 50000
SAVE_PATH = "./vibevoice_dataset/alvanlii_cantonese_youtube/checkpoint"

print("Loading raw dataset...")
features = Features({
    "id": Value("string"),
    "channel": Value("string"),
    "transcript_whisper": Value("string"),
    "title": Value("string"),
    "audio": Audio(sampling_rate=16000), # Standard for Vibe Voice / Whisper
    "transcript_sensevoice": Value("string"),
    "emotion_sensevoice": Sequence(Value("string")),
    "event_sensevoice": Sequence(Value("string")),
    "c50": Value("float32"),                # Force to string to prevent float error
    "snr": Value("float32"),                # Force to string
    "speech_duration": Value("float32"),    # Force to string
    "emotion_emotion2vec": Value("string")
})

def get_already_processed_count():
    # Check how many chunks we've already saved
    chunks = [d for d in os.listdir(SAVE_PATH) if d.startswith("chunk_")]
    return len(chunks) * CHUNK_SIZE

ds = load_from_disk("/root/VibeVoice/vibevoice_dataset/alvanlii_cantonese_radio")

sample = next(iter(ds))

print("\n--- First Sample Content ---")
print(f"Text Type: {type(sample)}")
print(f"Text Content: {sample['text']}")
print(f"Audio Type: {type(sample['audio'])}")
print(f"Audio Content: {sample['audio'].keys()}")

sf.write("preview.wav", sample['audio']["array"], sample['audio']["sampling_rate"])