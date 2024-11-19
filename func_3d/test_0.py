import numpy as np

def count_nonzero_elements(npy_file_path):
    data = np.load(npy_file_path)
    nonzero_count = np.count_nonzero(data)
    return nonzero_count

if __name__ == "__main__":
    npy_file_path = '/staff/wangtiantong/SAM2/data/acdc/Training/mask/patient001_frame01/patient001_frame01_seg_patch_2_label.npy'  # Replace with your actual npy file path
    count = count_nonzero_elements(npy_file_path)
    print(f"Number of non-zero elements: {count}")