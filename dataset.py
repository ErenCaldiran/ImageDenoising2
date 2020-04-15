import torch.utils.data as td
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from scipy.io import loadmat
import os
import numpy as np
from PIL import Image

class BSDS(td.Dataset):                     #Images are either 481x321 or 321x481. Make images 321x321

    def __init__(self, root_dir):
        super(BSDS, self).__init__()
        self.images_dir = os.path.join(root_dir)
        self.files = os.listdir(self.images_dir)

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        return "BSDS(mode={}, image_size={}, sigma={})"

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.files[idx])
        # random crop
        # i = np.random.randint(clean.size[0] - self.image_size[0])
        # j = np.random.randint(clean.size[1] - self.image_size[1])

        # clean = clean.crop([i, j, i+self.image_size[0], j+self.image_size[1]])
        transf = transforms.ToTensor()
        clean = transf(Image.open(img_path).convert('RGB'))

        return clean

class BSDS2(Dataset):
    def __init__(self, rootDirImg, rootDirGt):

        super(BSDS2, self).__init__()
        self.rootDirImg = os.path.join(rootDirImg)
        self.rootDirGt = os.path.join(rootDirGt)
        self.files = [os.listdir(self.rootDirImg), os.listdir(self.rootDirGt)]


        #self.rootDirImg = rootDirImg
        #self.rootDirGt = rootDirGt
        #self.listData = [sorted(os.listdir(rootDirImg)), sorted(os.listdir(rootDirGt))]
        #self.processed = processed

    def __len__(self):
        return len(self.files[1])

    def __getitem__(self, i):
        # input and target images
        inputImage = os.path.join(self.rootDirImg, self.files[i])
        targetImage = os.path.join(self.rootDirGt, self.files[i])

        #targetName = self.listData[1][i]
        # process the images

        transf = transforms.ToTensor()
        inputImage = transf(Image.open(inputImage).convert('RGB'))
        targetImage = transf(Image.open(targetImage).convert('L'))
        targetImage = (targetImage > 0.41).float()

        img = Image.fromarray(inputImage.numpy(), mode='RGB')
        target = Image.fromarray(targetImage.numpy(), mode='RGB')

        return img, target
