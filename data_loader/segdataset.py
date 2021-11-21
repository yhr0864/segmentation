from PIL import Image
import torch


class Data(torch.utils.data.Dataset):
    def __init__(self, datatxt: str, transform=None, target_transform=None):
        self.imgs = []
        with open(datatxt, "r") as fh:
            for line in fh:
                words = line.rstrip().split()
                self.imgs.append((words[0], words[1]))

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index: int):
        fn, label = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        label = Image.open(label).convert('L')

        if self.transform is not None:
            img = self.transform(img)
            label = self.transform(label)
        return img, label

    def __len__(self):
        return len(self.imgs)
