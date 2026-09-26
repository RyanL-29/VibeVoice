from datasets import load_dataset, Features, Value, Audio, Dataset, load_from_disk, concatenate_datasets
import asyncio
import io
import os
import numpy as np
import soundfile as sf
import edge_tts
from tqdm import tqdm

CHUNK_SIZE = 1000
CHUNKS_PER_RUN = 10                 # chunks generated per run (1 = 1000 rows); rerun to continue. None = all
TOTAL_RECORDS = 51148              # full dataset size; lower it to generate a subset
VOICE = "zh-HK-HiuGaaiNeural"
RATE = "+0%"                       # edge-tts speaking rate, e.g. "-10%" / "+10%"
SAMPLING_RATE = 24000              # edge-tts outputs 24kHz mono MP3, same rate VibeVoice trains on
MAX_CONCURRENCY = 8                # parallel edge-tts requests
MAX_RETRIES = 5
SAVE_PATH = "./vibevoice_dataset/zoengjyutgaai/lukdinggei/checkpoint"
FINAL_PATH = "./vibevoice_dataset/zoengjyutgaai/lukdinggei"

features = Features({
    "text": Value("string"),
    "audio": Audio(sampling_rate=SAMPLING_RATE),
})


def get_processed_chunks():
    os.makedirs(SAVE_PATH, exist_ok=True)
    return {d for d in os.listdir(SAVE_PATH) if d.startswith("chunk_")}


async def synthesize(text, semaphore):
    """Generate Cantonese speech for `text` and return a float32 numpy array (or None on failure)."""
    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                communicate = edge_tts.Communicate(text, VOICE, rate=RATE)
                mp3_bytes = bytearray()
                async for chunk in communicate.stream():
                    if chunk["type"] == "audio":
                        mp3_bytes.extend(chunk["data"])
                if not mp3_bytes:
                    raise RuntimeError("edge-tts returned no audio")

                array, sr = sf.read(io.BytesIO(bytes(mp3_bytes)), dtype="float32")
                if array.ndim > 1:
                    array = array.mean(axis=1)
                if sr != SAMPLING_RATE:
                    import librosa
                    array = librosa.resample(array, orig_sr=sr, target_sr=SAMPLING_RATE)
                return array.astype(np.float32)
            except Exception as e:
                if attempt == MAX_RETRIES:
                    print(f"\nFailed after {MAX_RETRIES} attempts: {e} | text: {text[:40]}...")
                    return None
                await asyncio.sleep(2 ** attempt)


async def generate_chunk(texts):
    semaphore = asyncio.Semaphore(MAX_CONCURRENCY)
    progress = tqdm(total=len(texts), leave=False)

    async def run(t):
        result = await synthesize(t, semaphore)
        progress.update(1)
        return result

    results = await asyncio.gather(*(run(t) for t in texts))
    progress.close()
    return results


def format_for_vibe(text, array):
    return {
        "text": f"Speaker 1: {text}",
        "audio": {"array": array, "sampling_rate": SAMPLING_RATE},
    }


print("Loading raw dataset...")
raw_ds = load_dataset("CanCLID/zoengjyutgaai", split="lukdinggei")
texts = raw_ds["transcription"]
raw_ds = raw_ds.select([i for i, t in enumerate(texts) if t and t.strip()])
raw_ds = raw_ds.rename_column("transcription", "text")
total = min(TOTAL_RECORDS, len(raw_ds))
print(f"{len(raw_ds)} usable rows, generating {total} with voice {VOICE}")

done_chunks = get_processed_chunks()
chunks_this_run = 0

for i in range(0, total, CHUNK_SIZE):
    chunk_name = f"chunk_{i:06d}"
    if chunk_name in done_chunks:
        continue
    if CHUNKS_PER_RUN is not None and chunks_this_run >= CHUNKS_PER_RUN:
        print(f"Reached {CHUNKS_PER_RUN} chunk(s) for this run; rerun the script to continue from record {i}")
        break
    chunks_this_run += 1

    end = min(i + CHUNK_SIZE, total)
    print(f"Processing records {i} to {end}...")
    texts = [t.strip() for t in raw_ds.select(range(i, end))["text"]]
    arrays = asyncio.run(generate_chunk(texts))

    chunk_data = [format_for_vibe(t, a) for t, a in zip(texts, arrays) if a is not None]
    print(f"{len(chunk_data)}/{len(texts)} clips generated")
    if not chunk_data:
        continue

    chunk_ds = Dataset.from_list(chunk_data, features=features)
    chunk_ds.save_to_disk(f"{SAVE_PATH}/{chunk_name}")
    print(f"{chunk_name} saved!")

done_count = len(get_processed_chunks())
total_chunks = (total + CHUNK_SIZE - 1) // CHUNK_SIZE
print(f"Progress: {done_count}/{total_chunks} chunks done")

all_chunks = []
for folder in sorted(get_processed_chunks()):
    all_chunks.append(load_from_disk(f"{SAVE_PATH}/{folder}"))

final_dataset = concatenate_datasets(all_chunks)
final_dataset.save_to_disk(FINAL_PATH)
print(f"Saved {len(final_dataset)} samples to {FINAL_PATH}")
