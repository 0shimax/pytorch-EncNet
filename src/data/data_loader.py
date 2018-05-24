from pathlib import Path
from skimage.color import rgba2rgb
import numpy as np
from sklearn.model_selection import train_test_split
from skimage import io
import torch
from torchvision import datasets
from torch.utils.data import Dataset


class CamVidDataset(Dataset):
    def __init__(self, image_file_names, root_dir,
                 subset=False, transform=None):
        super().__init__()
        self.image_file_names = image_file_names
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.image_file_names)

    def __getitem__(self, idx):
        img_name = self.image_file_names[idx]

        image_path = Path(self.root_dir, img_name)
        label_path = Path(self.root_dir, img_name.replace('.png', '_L.npz'))

        image = io.imread(image_path)
        image = image.transpose(2, 0, 1)
        label = np.load(label_path)['data']

        image = image[:, ::4, ::4]
        return torch.FloatTensor(image), torch.LongTensor(label[::4, ::4])


def loader(dataset, batch_size,  shuffle=True):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4)
    return loader

# def set_data_loader(args, train=True):
#     dataset = load_data(args)
#     train_data, test_data = train_test_split(dataset, test_size=.2)
#
#     assert train_data
#     assert test_data
#
#     if train:
#         dataset = train_data
#     else:
#         dataset = test_data
#
#     data_loader = torch.utils.data.DataLoader(
#         dataset, batch_size=args.batch_size,
#         shuffle=True, num_workers=int(args.workers))
#     return data_loader
