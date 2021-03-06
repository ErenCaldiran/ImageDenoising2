import os
import skimage
from skimage.metrics import peak_signal_noise_ratio
import torch
from torch import nn
from torch.autograd import Variable

from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import BSDS
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from matplotlib import pyplot as plt
import plotly.graph_objs as go
import plotly.offline as py
import torch.nn.functional as F


num_epochs = 301
batch_size = 32
learning_rate = 1e-3

rootDirImgTrain = "BSDS500/data/images/edit_train_test_mix/"
rootDirImgVal = "BSDS500/data/images/edited_val/"

train_set = BSDS(root_dir=rootDirImgTrain)
val_set = BSDS(root_dir=rootDirImgVal)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True) #pin_memory=True
valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

print(len(train_loader))
print(len(valid_loader))


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=24,      #3x 321x321 to  24x 161x161
                               kernel_size=3,stride=2,padding=1)
        self.bn1=nn.BatchNorm2d(24)

        self.conv2 = nn.Conv2d(in_channels=24,out_channels=48,     #24x 161x161 to 48x 81x81
                               kernel_size=3,stride=2,padding=1)
        self.bn2=nn.BatchNorm2d(48)

        self.conv3 = nn.Conv2d(in_channels=48, out_channels=96,    #48x 81x81 to 96x 41x41
                               kernel_size=3, stride=2, padding=1)
        self.bn3=nn.BatchNorm2d(96)

        self.conv4 = nn.Conv2d(in_channels=96, out_channels=128,    #96x 41x41 to 128x 21x21
                               kernel_size=3, stride=2, padding=1)
        self.bn4=nn.BatchNorm2d(128)

        self.conv_t1 = nn.ConvTranspose2d(in_channels=128,out_channels=96,kernel_size=3,stride=2,padding=1)  #128x 21x21 to 96x 41x41

        self.conv_t2 = nn.ConvTranspose2d(in_channels=96,out_channels=48,kernel_size=3,stride=2,padding=1)   #96x 41x41 to 48x 81x81

        self.conv_t3 = nn.ConvTranspose2d(in_channels=48,out_channels=24,kernel_size=3,stride=2, padding=1)  #48x 81x81 to 24x 161x161

        self.conv_t4 = nn.ConvTranspose2d(in_channels=24,out_channels=3,kernel_size=3,stride=2, padding=1)   #24x 161x161 to 3x 321x321

    def forward(self, x):
        x = torch.tanh(self.bn1(self.conv1(x)))
        x = torch.tanh(self.bn2(self.conv2(x)))
        x = torch.tanh(self.bn3(self.conv3(x)))
        x = torch.tanh(self.bn4(self.conv4(x)))
        x = x.view(-1, 128, 21, 21)  # reshape back to feature map format
        x = torch.tanh(self.conv_t1(x))
        x = torch.tanh(self.conv_t2(x))
        x = torch.tanh(self.conv_t3(x))
        x = torch.tanh(self.conv_t4(x))
        return x

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AutoEncoder().to(device)
criterion = nn.MSELoss().to(device)
optimizer = torch.optim.Adam(
    model.parameters(), lr=learning_rate) #Weight decay used to prevent oscillations, 4e-3 is ideal.


def train():
    model.train().cuda() # put  in train mode #.to(device) in laptop
    total_loss = torch.zeros(1)
    for img in train_loader:  # next batch
        img = Variable(img).to(device) # convert to Variable to calculate gradient and move to gpu
        gaussian_img = skimage.util.random_noise(img.cpu(), mode="gaussian", var=0.04)
        gaussian_img = torch.from_numpy(gaussian_img).to(device)
        #saltpepper_img = skimage.util.random_noise(img.cpu(), mode="s&p", amount=0.10)
        #saltpepper_img = torch.from_numpy(saltpepper_img).to(device)
        img_ndarr = (img.cpu()).numpy()

        optimizer.zero_grad()
        #torch.cuda.empty_cache()

        output = model(gaussian_img.float()).to(device) # feed forward

        loss = criterion(output, img)    # calculate loss
        output_ndarr = (output.cpu().detach()).numpy()
        psnr = peak_signal_noise_ratio(img_ndarr,output_ndarr)
        loss.backward()  # calculate new gradients
        optimizer.step()  # update weights
        total_loss += loss.item()  # accumulate loss

    return gaussian_img,img, output, total_loss, psnr

def valid():
    with torch.no_grad():
        model.eval().cuda()
        valid_loss = torch.zeros(1)
        for img in valid_loader:
            img = Variable(img).to(device)  # convert to Variable to calculate gradient and move to gpu
            img_ndarr = (img.cpu()).numpy()
            gaussian_image = skimage.util.random_noise(img.cpu(), mode="gaussian", var=0.04) #default 0.01
            gaussian_image = torch.from_numpy(gaussian_image).to(device)
            #saltpepper_img = skimage.util.random_noise(img.cpu(), mode="s&p", amount=0.10)
            #saltpepper_img = torch.from_numpy(saltpepper_img).to(device)
            # image, labels = image.to(device), labels.to(device)
            output = model(gaussian_image.float()).to(device)
            valid_loss += criterion(output, img)  # calculate loss
            output_ndarr = (output.cpu().detach()).numpy()
            psnr = peak_signal_noise_ratio(img_ndarr, output_ndarr)

        return  gaussian_image,img, output, valid_loss, psnr

epocharray = []
trainlossarray = []
validlossarray = []
trainsnr = []
validsnr = []
inTotalData = 0


for epoch in range(num_epochs):
    noised_img,img, output, loss, psnr = train()
    valid_noised_img,valid_img, valid_output, valid_loss, valid_psnr = valid()

    epocharray.append(epoch)
    trainlossarray.append(loss.item() / len(train_loader))
    validlossarray.append(valid_loss.item() / len(valid_loader))
    trainsnr.append((psnr))
    validsnr.append((valid_psnr))


    print('epoch [{}/{}], loss:{:.4f}, SNR:{}'
        .format(epoch + 1, num_epochs, loss.item()/len(train_loader), psnr))
    print('Validation_loss:{}, SNR: {}'
          .format(valid_loss.item() / len(valid_loader), valid_psnr))

    if epoch % 30 == 0:
        #torch.cuda.empty_cache()
        pic_org = (img)
        pic_noised = (noised_img)
        pic_pred = (output)
        save_image(pic_org, './denoise_image_org__{}.png'.format(epoch))
        save_image(pic_noised, './denoise_image_noised__{}.png'.format(epoch))
        save_image(pic_pred, './denoise_image_pred__{}.png'.format(epoch))

        valid_org = (valid_img)
        valid_noisy = (valid_noised_img)
        valid_pic = (valid_output.cpu().data)
        save_image(valid_pic, './valid_denoise_image_pred{}.png'.format((epoch)))
        save_image(valid_noisy, './valid_denoise_image_noise_{}.png'.format((epoch)))
        save_image(valid_org, './valid_denoise_image_org_{}.png'.format((epoch)))


trainErr = go.Scatter(x=epocharray,
                            y=trainlossarray,
                            name = "Train loss",
                            marker={'color': 'blue', 'symbol': 100, 'size': 3},
                            mode="lines")

validErr = go.Scatter(x=epocharray,
                            y=validlossarray,
                            name = "Valid loss",
                            marker={'color': 'red', 'symbol': 100, 'size': 3},
                            mode="lines")

inTotalData = [trainErr,validErr]

layout = dict(title = 'Train and validation loss',
              xaxis = dict(title = 'Epoch'),
              yaxis = dict(title = 'Loss'),
              )

InTotalfigure = dict(data=inTotalData, layout=layout)

py.plot(InTotalfigure, filename='NoNoiseFirstTryLoss.html', show_link=True)


trainSNR = go.Scatter(x=epocharray,
                           y=trainsnr,
                           name = "Train snr",
                           marker={'color': 'blue', 'symbol': 100, 'size': 3},
                           mode="lines")

validSNR = go.Scatter(x=epocharray,
                      y=validsnr,
                      name="Valid snr",
                      marker={'color': 'red', 'symbol': 100, 'size': 3},
                      mode="lines")

TotalData = [trainSNR,validSNR]

SNRlayout = dict(title = 'Train and validation snr',
              xaxis = dict(title = 'Epoch'),
              yaxis = dict(title = 'Loss'),
              )

Totalfigure = dict(data=TotalData, layout=SNRlayout)

py.plot(Totalfigure, filename='NoNoiseFirstTrySnr.html', show_link=True)

torch.save(model.state_dict(), './conv_autoencoder.pth')