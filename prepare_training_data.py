import os
import shutil

src_dir = 'data/processed_backup'
dst_dir = 'data/processed'

words_to_train = [
    'look', 'shhh', 'donkey', 'hear', 'bird',
    'fireman', 'eye', 'mom', 'gift', 'horse',
    'hello', 'alligator', 'see', 'carrot', 'all',
    'hair', 'hungry', 'elephant', 'fall', 'wet',
    'minemy', 'like', 'empty', 'white', 'will',
    'now', 'refrigerator', 'puppy', 'bed', 'after',
    'have', 'person', 'hat', 'beside'
]

if os.path.exists(dst_dir) and not os.path.exists(src_dir):
    print("Backing up data/processed to data/processed_backup...")
    os.rename(dst_dir, src_dir)
elif not os.path.exists(src_dir):
    print("ERROR: No backup or processed directory found.")
    exit(1)

os.makedirs(dst_dir, exist_ok=True)

counts = {w: 0 for w in words_to_train}
for f in os.listdir(src_dir):
    if f.endswith('.npy'):
        word = f.split('_')[0]
        if word in words_to_train:
            shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))
            counts[word] += 1

print("Extracted counts per word:")
for w, c in counts.items():
    status = "✅" if c > 0 else "❌ NOT FOUND"
    print(f"  {status} {w}: {c} samples")
print(f"\nTotal: {sum(counts.values())} sequences across {sum(1 for c in counts.values() if c > 0)} words")
