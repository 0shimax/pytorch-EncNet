from sklearn.model_selection import train_test_split
import torch
from torch import transforms
from torchvision import datasets
from torch.utils.data import Dataset


class CamVidDataset(Dataset):
    def __init__(self, label_paths, root_dir, subset=False, transform=None):
        self.label_paths = label_paths
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_name = self.labels.iloc[idx, 0] # file name
        fullname = join(self.root_dir, img_name)
        image = Image.open(fullname).convert('RGB')
        labels = self.labels.iloc[idx, 2] # category_id
        if self.transform:
            image = self.transform(image)
        return image, labels


def load_data(args):
    dataset = \
        datasets.ImageFolder(root=args.dataroot,
                             transform=transforms.Compose([
                                 transforms.Resize(args.image_size),
                                 transforms.ColorJitter(brightness=0.4,
                                                        contrast=0.4,
                                                        saturation=0.4),
                                 # transforms.CenterCrop(opt.imageSize),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize(
                                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                             ]))
    return dataset


def set_data_loader(args, train=True):
    dataset = load_data(args)
    train_data, test_data = train_test_split(dataset, test_size=.2)

    assert train_data
    assert test_data

    if train:
        dataset = train_data
    else:
        dataset = test_data

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=True, num_workers=int(args.workers))
    return data_loader
