import torch
from torch.utils.data import Dataset
import PIL.Image as Image
import os
import torchvision.datasets as dset
from torchvision.transforms import transforms
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
def make_dataset(root):
    imgs_sick=[]
    imgs_healthy=[]
    imgs = []
    train_dataset = dset.ImageFolder(root)
    for i in train_dataset.imgs:
        if i[1] == 0:
            img = os.path.join(i[0])
            imgs_healthy.append((img, 0))
        else:
            img = os.path.join(i[0])
            imgs_sick.append((img, 1))
    n_healthy = len(imgs_healthy)//2
    n_sick = len(imgs_sick)//2
    for i in range(n_healthy):
        img=os.path.join(root + '/healthy',"healthy.%03d.jpg"%i)
        mask=os.path.join(root + '/healthy',"healthy.%03d_mask.jpg"%i)
        label = 0
        imgs.append((img, mask, label))
    for j in range(n_sick):
        img=os.path.join(root + '/sick',"sick.%03d.jpg"%j)
        mask=os.path.join(root + '/sick',"sick.%03d_mask.jpg"%j)
        label = 1
        imgs.append((img, mask, label))
    return imgs




class LiverDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_dataset(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path, label = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        if self.transform is not None:
            img_x = self.transform(img_x)
        if self.target_transform is not None:
            img_y = self.target_transform(img_y)
        return img_x, img_y, label

    def __len__(self):
        return len(self.imgs)


liver_dataset = LiverDataset("data/healthysick_2",transform=x_transforms,target_transform=y_transforms)
print(liver_dataset.__getitem__(2).shape)