import argparse
from pathlib import Path
import torch
import torch.optim as optim
import torch.nn.functional as F

from model.resnet18 import resnet18
from model.utils import calculate_l1_loss
from data.data_loader import set_data_loader

torch.manual_seed(555)


def main(args):
    model = resnet18()
    critic = torch.nn.NLLLoss2D()

    # setup optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
    # optimizer = optim.RMSprop(model.parameters(), lr=0.001)

    train_data_loader = set_data_loader(args, train=True)
    test_data_loader = set_data_loader(args, train=False)


def train(args, model, optimizer, data_loader):
    model.train()
    for epoch in range(args.epochs):
        for i, data in enumerate(data_loader, 0):
            model.zero_grad()
            data, target = data['sample'], data['target']

            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss2D(output, target)
            loss += calculate_l1_loss(output, target)

            loss.backward()
            optimizer.step()
            print('[{}/{}][{}/{}] Loss: {:.4f}'.format(
                  epoch, args.epoch, i,
                  len(data_loader), loss.item()))

        # do checkpointing
        torch.save(model.state_dict(),
                   '{}/encnet_ckpt.pth'.format(args.out_dir))


def test(args, model, data_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data in data_loader:
            data, target = data['sample'], data['target']
            output = model(data)
            # sum up batch loss
            test_loss += F.nll_loss2D(
                output, target, size_average=False).item()
            # get the index of the max log-probability
            pred = output.max(1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(data_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
          .format(test_loss, correct, len(data_loader.dataset),
                  100. * correct / len(data_loader.dataset)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataroot', default='./data', help='path to dataset')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
    parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
    parser.add_argument('--epochs', type=int, default=200, help='number of epochs to train for')
    parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
    parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
    parser.add_argument('--outf', default='./results', help='folder to output images and model checkpoints')
    args = parser.parse_args()

    main(args)
