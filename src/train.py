import argparse
from pathlib import Path
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms

# from model.resnet18 import resnet18
from model.enc_net import enc_net
from model.utils import calculate_l1_loss
from data.data_loader import CamVidDataset, loader


torch.manual_seed(555)


def main(args):
    model = enc_net(32)

    # setup optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    # optimizer = optim.RMSprop(model.parameters(), lr=0.001)

    train_image_names =\
        [line.rstrip() for line in open(args.train_image_pointer_path)]
    test_image_names =\
        [line.rstrip() for line in open(args.test_image_pointer_path)]

    resize_shape=(240, 180)

    train_dataset = CamVidDataset(train_image_names, args.root_dir)
    test_dataset = CamVidDataset(test_image_names, args.root_dir)
    train_loader = loader(train_dataset, args.batch_size)
    test_loader = loader(test_dataset, args.batch_size, shuffle=False)

    # train(args, model, optimizer, train_loader)
    test(args, model, test_loader)


def train(args, model, optimizer, data_loader):
    model.train()
    for epoch in range(args.epochs):
        for i, (data, target) in enumerate(data_loader):
            model.zero_grad()

            optimizer.zero_grad()
            output, se2, se1 = model(data)
            n_batch = output.shape[0]
            loss = torch.mean(F.nll_loss(F.log_softmax(output), target))
            loss += torch.mean(calculate_l1_loss(output, target))

            exist_class = [[1 if c in target[i_batch].numpy() else 0 for c in range(32)]
                            for i_batch in range(n_batch)]
            exist_class = torch.FloatTensor(exist_class)

            loss += torch.mean(F.mse_loss(se2, exist_class))
            loss += torch.mean(F.mse_loss(se1, exist_class))

            loss.backward()
            optimizer.step()
            print('[{}/{}][{}/{}] Loss: {:.4f}'.format(
                  epoch, args.epochs, i,
                  len(data_loader), loss.item()))

        # do checkpointing
        torch.save(model.state_dict(),
                   '{}/encnet_ckpt.pth'.format(args.out_dir))


def test(args, model, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in data_loader:
            output, se2, se1 = model(data)
            # sum up batch loss
            test_loss += torch.mean(F.nll_loss(
                output, target, size_average=False)).item()
            # get the index of the max log-probability
            pred = output.argmax(1)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          .format(test_loss, correct, len(data_loader.dataset),
                  100. * correct / len(data_loader.dataset)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', default='./data/CamSeq01', help='path to dataset')
    parser.add_argument('--train-image-pointer-path', default='./data/train_image_pointer', help='path to train image pointer')
    parser.add_argument('--test-image-pointer-path', default='./data/test_image_pointer', help='path to test image pointer')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch-size', type=int, default=4, help='input batch size')
    parser.add_argument('--image-size', type=int, default=256, help='the height / width of the input image to network')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--out-dir', default='./results', help='folder to output images and model checkpoints')
    args = parser.parse_args()
    Path(args.out_dir).mkdir(parents=True, exist_ok=True),

    main(args)
