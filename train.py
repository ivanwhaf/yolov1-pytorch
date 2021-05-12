import argparse
import os
import time

import matplotlib.pyplot as plt
import torch
from torch.optim import SGD
from torchvision import utils

from utils import create_dataloader, YOLOLoss, parse_cfg, build_model

# from torchviz import make_dot

parser = argparse.ArgumentParser(description='YOLOv1-pytorch')
parser.add_argument("--cfg", "-c", default="cfg/yolov1.cfg", help="Yolov1 config file path", type=str)
parser.add_argument("--dataset_cfg", "-d", default="cfg/dataset.cfg", help="Dataset config file path", type=str)
parser.add_argument("--weights", "-w", default="", help="Pretrained model weights path", type=str)
parser.add_argument("--output", "-o", default="output", help="Output path", type=str)
parser.add_argument("--epochs", "-e", default=100, help="Training epochs", type=int)
parser.add_argument("--lr", "-lr", default=0.001, help="Training learning rate", type=float)
parser.add_argument("--batch_size", "-bs", default=16, help="Training batch size", type=int)
parser.add_argument("--input_size", "-is", default=448, help="Image input size", type=int)
parser.add_argument("--save_freq", "-sf", default=10, help="Frequency of saving model checkpoint when training",
                    type=int)
args = parser.parse_args()


def train(model, train_loader, optimizer, epoch, device, S, B, train_loss_lst):
    model.train()  # Set the module in training mode
    train_loss = 0
    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)

        # back prop
        criterion = YOLOLoss(S, B)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        # show batch0 dataset
        if batch_idx == 0 and epoch == 0:
            fig = plt.figure()
            inputs = inputs.cpu()  # convert to cpu
            grid = utils.make_grid(inputs)
            plt.imshow(grid.numpy().transpose((1, 2, 0)))
            plt.savefig(os.path.join(output_path, 'batch0.png'))
            # plt.show()
            plt.close(fig)

        # print loss and accuracy
        if batch_idx % 5 == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]  Loss: {:.6f}'
                  .format(epoch, batch_idx * len(inputs), len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), loss.item()))

    # record training loss
    train_loss /= len(train_loader)
    train_loss_lst.append(train_loss)
    return train_loss_lst


def validate(model, val_loader, device, S, B, val_loss_lst):
    model.eval()  # Sets the module in evaluation mode
    val_loss = 0
    # no need to calculate gradients
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # add one batch loss
            criterion = YOLOLoss(S, B)
            val_loss += criterion(output, target).item()

    val_loss /= len(val_loader)
    print('\nVal set: Average loss: {:.4f}'.format(val_loss))

    # record validating loss
    val_loss_lst.append(val_loss)
    return val_loss_lst


def test(model, test_loader, device, S, B):
    model.eval()  # Sets the module in evaluation mode
    test_loss = 0
    # no need to calculate gradients
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)

            # add one batch loss
            criterion = YOLOLoss(S, B)
            test_loss += criterion(output, target).item()

    # record testing loss
    test_loss /= len(test_loader)
    print('Test set: Average loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    cfg = parse_cfg(args.cfg)
    dataset_cfg = parse_cfg(args.dataset_cfg)
    img_path, label_path = dataset_cfg['images'], dataset_cfg['labels']
    S, B, num_classes = int(cfg['S']), int(cfg['B']), int(cfg['num_classes'])

    # create output file folder
    start = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
    output_path = os.path.join(args.output, start)
    os.makedirs(output_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build model
    model = build_model(args.weights, S, B, num_classes).to(device)

    # plot model structure
    # graph = make_dot(model(torch.rand(1, 3, args.input_size, args.input_size).cuda()),
    #                  params=dict(model.named_parameters()))
    # graph.render('model_structure', './', cleanup=True, format='png')

    # get data loader
    train_loader, val_loader, test_loader = create_dataloader(img_path, label_path, 0.8, 0.1, 0.1, args.batch_size,
                                                              args.input_size, S, B, num_classes)

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0.0005)
    # optimizer = Adam(model.parameters(), lr=lr)

    train_loss_lst, val_loss_lst = [], []

    # train epoch
    for epoch in range(args.epochs):
        train_loss_lst = train(model, train_loader, optimizer, epoch, device, S, B, train_loss_lst)
        val_loss_lst = validate(model, val_loader, device, S, B, val_loss_lst)

        # save model weight every save_freq epoch
        if epoch % args.save_freq == 0 and epoch >= args.epochs / 3:
            torch.save(model.state_dict(), os.path.join(output_path, 'epoch' + str(epoch) + '.pth'))

    test(model, test_loader, device, S, B)

    # save model
    torch.save(model.state_dict(), os.path.join(output_path, 'last.pth'))

    # plot loss, save params change
    fig = plt.figure()
    plt.plot(range(args.epochs), train_loss_lst, 'g', label='train loss')
    plt.plot(range(args.epochs), val_loss_lst, 'k', label='val loss')
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('acc-loss')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(output_path, 'loss_curve.jpg'))
    plt.show()
    plt.close(fig)
