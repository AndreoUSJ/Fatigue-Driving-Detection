import os
import shutil
from PIL import Image
import imagehash

INPUT_FOLDER = r"C:\Users\User\Downloads\Drowsy_dataset\val\NATURAL"
OUTPUT_FOLDER = r"C:\Users\User\Downloads\Drowsy_dataset\val\NEW_NATURAL"

HASH_SIZE = 8
MAX_HASH_DIFF = 3   # lower = stricter, 0 means exact same only

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

valid_exts = (".jpg", ".jpeg", ".png", ".bmp")
files = sorted([f for f in os.listdir(INPUT_FOLDER) if f.lower().endswith(valid_exts)])

kept_hashes = []
kept_count = 0
removed_count = 0

for filename in files:
    path = os.path.join(INPUT_FOLDER, filename)

    try:
        img = Image.open(path).convert("L")
    except Exception:
        print(f"Skipping unreadable file: {filename}")
        continue

    current_hash = imagehash.phash(img, hash_size=HASH_SIZE)

    too_similar = False
    for h in kept_hashes:
        if current_hash - h <= MAX_HASH_DIFF:
            too_similar = True
            break

    if too_similar:
        removed_count += 1
        print(f"Removed similar: {filename}")
    else:
        shutil.copy2(path, os.path.join(OUTPUT_FOLDER, filename))
        kept_hashes.append(current_hash)
        kept_count += 1
        print(f"Kept: {filename}")

print("\nDone.")
print(f"Kept: {kept_count}")
print(f"Removed: {removed_count}")