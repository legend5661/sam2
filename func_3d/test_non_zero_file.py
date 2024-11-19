import os
import numpy as np

def find_non_zero_npy_files(directory):
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                data = np.load(file_path)
                if np.any(data):
                    print(file_path)

if __name__ == "__main__":
    directory = "/staff/wangtiantong/SAM2/data/acdc/Training/mask/patient005_frame01"  # Replace with your directory path
    find_non_zero_npy_files(directory)