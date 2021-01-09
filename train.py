import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from classify import model_class
from dataset import LiverDataset
import torch.nn.functional as F
import numpy as np
# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_transforms = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# mask只需要转换为tensor
y_transforms = transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),
])


def train_model(model, criterion, optimizer, dataload, num_epochs=50):
    import matplotlib.pyplot as plt
    acc_arr = []
    loss_arr = []
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y, z in dataload:
            step += 1
            inputs = x.to(device)
            # labels = y.to(device)
            target = z
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            loss = criterion(outputs, target)
            print(loss)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            correct = 0
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            print('Accuracy: {}/{} ({:.0f}%)\n'.format(correct, 8,100. * correct / 8))
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        loss_arr.append(epoch_loss / step)
        with open('loss2.txt', 'a') as file_object:
            file_object.write(str(epoch))
            file_object.write(" ")
            file_object.write(str(epoch_loss / step))
            file_object.write("\n")
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss/step))
    torch.save(model.state_dict(), 'weights_%d.pth' % epoch)
    x1 = range(0, 30)
    x2 = range(0, 500)
    y1 = loss_arr
    y2 = acc_arr
    plt.subplot(2, 1, 1)
    plt.plot(x2, y2, '.-')
    plt.title('Test accuracy vs. epoches')
    plt.ylabel('Test accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x1, y1, '.-')
    plt.xlabel('Test loss vs. epoches')
    plt.ylabel('Test loss')
    plt.show()
    plt.savefig("accuracy_loss.jpg")
    return model

#训练模型
def train():
    model = model_class(3, 1).to(device)
    batch_size = 8
    print(batch_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(),lr=0.0001,weight_decay=0.0001)
    liver_dataset = LiverDataset("data/healthysick_2",transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    train_model(model, criterion, optimizer, dataloaders)


def test(args):
    model = model_class(3, 1)
    model.load_state_dict(torch.load(args.ckpt,map_location='cpu'))
    liver_dataset = LiverDataset("data/healthysick_2", transform=x_transforms,target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    model.eval()
    test_loss = 0
    correct = 0
    ak = correct
    import matplotlib.pyplot as plt
    import torchvision.utils as vutils
    plt.ion()
    right_arr=[]
    with torch.no_grad():
        for x, y, target in dataloaders:
            output1= model(x)
            test_loss += F.nll_loss(output1, target, reduction='sum').item()
            pred = output1.argmax(dim=1, keepdim=True)
            print("pretend: {}".format(pred.view_as(target)))
            print('target:  {}'.format(target))
            correct += pred.eq(target.view_as(pred)).sum().item()
            print("-----------")
    test_loss /= len(liver_dataset)
    print('Average loss is: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct, len(liver_dataset), 100.*correct/len(liver_dataset)))


if __name__ == '__main__':
    #参数解析
    parse=argparse.ArgumentParser()
    parse = argparse.ArgumentParser()
    parse.add_argument("--action", type=str, help="train or test")
    parse.add_argument("--batch_size", type=int, default=8)
    parse.add_argument("--ckpt", type=str, help="the path of model weight file")
    args = parse.parse_args()

    if args.action=="train":
        train()
    elif args.action=="test":
        test(args)








