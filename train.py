import os
import time
import torch
import argparse
from torch import nn
from model import YOLOv1
import matplotlib.pyplot as plt
from torchvision import utils
from torch.optim import SGD, Adam

from utils.util import YOLOLoss, parse_cfg
from utils.datasets import create_dataloader


def train(net, train_loader, optimizer, epoch, device, train_loss_lst):
    model.train()  # Set the module in training mode
    train_loss = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        # inputs, labels = inputs.to(device), labels.to(device)
        # x = torch.randn(1, 3, 448, 448)
        # label = torch.randn(1, 7, 7, 30)
        outputs = model(inputs)
        # print(output)

        optimizer.zero_grad()
        loss = criterion(outputs, labels)
        train_loss += loss
        loss.backward()
        optimizer.step()
        print(loss)

        # show batch0 dataset
        if batch_idx == 0 and epoch == 0:
            fig = plt.figure()
            inputs = inputs.cpu()  # convert to cpu
            grid = utils.make_grid(inputs)
            plt.imshow(grid.numpy().transpose((1, 2, 0)))
            plt.show()

        # print loss and accuracy
        if (batch_idx+1) % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]  Loss: {:.6f}'
                  .format(epoch, batch_idx * len(inputs), len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader.dataset)
    # record loss
    train_loss_lst.append(train_loss)
    return train_loss_lst


def validate(model, val_loader, device, val_loss_lst):
    model.eval()  # Sets the module in evaluation mode
    val_loss = 0
    # no need to calculate gradients
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # add one batch loss
            criterion = nn.CrossEntropyLoss()
            val_loss += criterion(output, target)
            # val_loss += F.nll_loss(output, target, reduction='sum').item()

            # find index of max prob
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(val_loader.dataset)
    print('\nVal set: Average loss: {:.4f}'.format(val_loss))

    # record loss
    val_loss_lst.append(val_loss)
    return val_loss_lst


def test(model, test_loader, device):
    model.eval()  # Sets the module in evaluation mode
    test_loss = 0
    # no need to calculate gradients
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # add one batch loss
            criterion = nn.CrossEntropyLoss()
            test_loss += criterion(output, target)
            # test_loss += F.nll_loss(output, target, reduction='sum').item()

            # find index of max prob
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

    # record loss and acc
    test_loss /= len(test_loader.dataset)
    print('Test set: Average loss: {:.4f}'.format(test_loss))


def arg_parse():
    """
    Parse arguements to the detect module

    """

    parser = argparse.ArgumentParser(description='Food Recognition System')

    parser.add_argument("--cfg", "-c", dest='cfg', default="cfg/yolov1.cfg",
                        help="Your yolo config file path", type=str)

    parser.add_argument("--data", "-d", dest='data', default="cfg/dataset.cfg",
                        help="Your dataset config file path", type=str)

    parser.add_argument("--weights", "-w", dest='weights', default="weights/yolov1.pth",
                        help="Path of pretrained weights", type=str)

    parser.add_argument("--epochs", "-e", dest='epochs', default=100,
                        help="Training epochs", type=int)

    parser.add_argument("--lr", "-lr", dest='lr', default=0.005,
                        help="Training learning rate", type=float)

    parser.add_argument("--batch_size", "-b", dest='batch_size', default=16,
                        help="Training batch size", type=int)

    parser.add_argument("--input_size", "-i", dest='input_size', default=448,
                        help="Image input size", type=int)

    parser.add_argument("--save_freq", "-s", dest='save_freq', default=10,
                        help="Frequence of saving model checkpoints when training", type=int)

    return parser.parse_args()


if __name__ == "__main__":
    # torch.manual_seed(0)
    args = arg_parse()
    cfg, data, weights, epochs, lr, batch_size, input_size, save_freq = args.cfg, args.data, args.weights, args.epochs, args.lr, args.batch_size, args.input_size, args.save_freq
    cfg = parse_cfg(cfg)
    data_cfg = parse_cfg(data)
    img_path, label_path = data_cfg['dataset'], data_cfg['label']

    # load dataset and dataloader
    train_loader, val_loader, test_loader = create_dataloader(img_path, label_path,
                                                              0.8, 0.1, 0.1, batch_size, input_size)

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device="cpu"

    # build model
    model = YOLOv1(B=2, nb_classes=10)

    optimizer = SGD(model.parameters(), lr=lr, momentum=0.9)
    criterion = YOLOLoss()

    # create output file folder
    start = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    os.makedirs('weights/'+start)

    # loss and list
    train_loss_lst, val_loss_lst = [], []

    for epoch in range(epochs):
        train_loss_lst = train(model, train_loader,
                               optimizer, epoch, device, train_loss_lst)
        # val_loss_lst = validate(model, val_loader, device, val_loss_lst)

        # save model weights every save_freq epoch
        if epoch % save_freq == 0:
            torch.save(model.state_dict(), os.path.join(
                'weights', start, 'epoch'+str(epoch)+'.pth'))

        # end for batch
    # end for epoch

    # test(model, test_loader, device)

    # plot loss and accuracy, save params change
    fig = plt.figure()
    plt.plot(range(epochs), train_loss_lst, 'g', label='train loss')
    plt.plot(range(epochs), val_loss_lst, 'k', label='val loss')
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('acc-loss')
    plt.legend(loc="upper right")
    now = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    plt.savefig('./visualize/parameter/' + now + '.jpg')
    plt.show()

    # save model
    torch.save(model.state_dict(), os.path.join('weights', start, 'last.pth'))
