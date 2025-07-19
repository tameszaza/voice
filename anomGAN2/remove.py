import os
import glob

root_dir = '../data_wavefake'  # change this to your target directory

# Recursively find all .npy files
npy_files = glob.glob(os.path.join(root_dir, '**', '*.npy'), recursive=True)

# Delete each file
for file_path in npy_files:
    try:
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    except Exception as e:
        print(f"Failed to delete {file_path}: {e}")
