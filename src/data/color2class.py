import os
from pathlib import Path
import pandas as pd
import numpy as np
from skimage import io
from matplotlib import pyplot as plt


data_root_dir = './data'
label_table_name = 'label_color'
image_dir = 'CamSeq01'
image_dir_path = Path(data_root_dir, image_dir)

label_table = pd.read_csv(Path(data_root_dir, label_table_name))
label_image_file_names =\
    [f for f in os.listdir(image_dir_path) if '_L.png' in f]

for image_file_name in label_image_file_names:
    file_path = Path(image_dir_path, image_file_name)
    print('processing {}'.format(file_path))
    img = io.imread(file_path)
    # plt.figure()
    # io.imshow(img)
    # plt.show()

    class_map = np.empty((img.shape[0], img.shape[1]), dtype='i')
    for i_row in range(len(label_table)):
        class_color = label_table.iloc[i_row]
        idx = np.where(
            (img[:, :, 0] == class_color[0])  # r
            & (img[:, :, 1] == class_color[1])  # g
            & (img[:, :, 2] == class_color[2]))  # b
        class_map[idx] = i_row
    label_file_name = Path(image_dir_path,
                           os.path.splitext(image_file_name)[0]+'.npz')
    np.savez(label_file_name, data=class_map)
