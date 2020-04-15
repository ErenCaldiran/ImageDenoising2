import os
import skimage
from skimage.metrics import peak_signal_noise_ratio
import torch
from torch import nn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset import BSDS, BSDS2
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.utils import save_image
from matplotlib import pyplot as plt
import plotly.graph_objs as go
import plotly.offline as py


num_epochs = 301
batch_size = 40
learning_rate = 1e-3

rootDirImgTrain = "BSDS500/data/images/edit_train_test_mix/"
rootDirImgVal = "BSDS500/data/images/edited_val/"

train_set = BSDS(root_dir=rootDirImgTrain)
val_set = BSDS(root_dir=rootDirImgVal)

train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True) #pin_memory=True
valid_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

print(len(train_loader))
print(len(valid_loader))

#trainDS = BSDS2(rootDirImgTrain, rootDirGtTrain)
#valDS = BSDS2(rootDirImgVal, rootDirGtVal, preprocessed)
#trainDS = ConcatDataset([trainDS,valDS])

#train_loader2 = DataLoader(trainDS, shuffle=True, batch_size=1, num_workers=4)

#print("22222aaa")
#print(len(train_loader2))


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()                     #starts with 321x321
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=32,       #32x 161x161   with kernel 5 32x 159x159
                               kernel_size=5,stride=2,padding=1)
        self.bn1=nn.BatchNorm2d(32)
                                                                    #64x 81x81                    64x 79x79
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,
                               kernel_size=5,stride=2,padding=1)
        self.bn2=nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128,    #128x 41x41                    128x 39x39
                               kernel_size=5, stride=2, padding=1)
        self.bn3=nn.BatchNorm2d(128)

        # first fully connected layer from 128x41x41=215.168 input features to 1691 hidden units   divided by 128
        self.fc11 = nn.Linear(in_features=128*39*39,out_features=1521)

        self.fc1 = nn.Linear(in_features=1521,out_features=39)  #square rooted     #The formula for convolutional layers : Width = [ (Width – KernelWidth + 2*Padding) / Stride] + 1.

        self.fc2 = nn.Linear(in_features=39,out_features=1521)

        self.fc22 = nn.Linear(in_features=1521,out_features=128*39*39)      #The formula for transpose convolutional layers: Wout = stride(Win - 1) + kernelsize – 2*padding

        self.conv_t1 = nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=5,    #64x 79x79
                                          stride=2,padding=1)

        self.conv_t2 = nn.ConvTranspose2d(in_channels=64,out_channels=32,kernel_size=5,     #32x 159x159
                                          stride=2,padding=1)

        self.conv_t3 = nn.ConvTranspose2d(in_channels=32,out_channels=3,kernel_size=5,     #3x 321x321
                                          stride=2)

    def forward(self, x):
        x = torch.tanh(self.bn1(self.conv1(x)))
        x = torch.tanh(self.bn2(self.conv2(x)))
        x = torch.tanh(self.bn3(self.conv3(x)))
        x = torch.flatten(x, 1)  # flatten feature maps, Bx (CxHxW)
        x = torch.tanh(self.fc11(x))
        x = torch.tanh(self.fc1(x))                                         #formula for Maxpool(Uses floor as default) :Hout= [ (Hin  + 2∗padding – dilation × (kernel_size − 1) – 1)/stride]
        x = torch.tanh(self.fc2(x))                                         #Wout= [ (Win  + 2∗padding – dilation × (kernel_size − 1) – 1)/stride]
        x = torch.tanh(self.fc22(x))
        x = x.view(-1, 128, 39, 39)  # reshape back to feature map format
        x = torch.tanh(self.conv_t1(x))
        x = torch.tanh(self.conv_t2(x))
        x = torch.tanh(self.conv_t3(x))
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
        #gaussian_img = skimage.util.random_noise(img.cpu(), mode="gaussian", var=1.6)
        #gaussian_img = torch.from_numpy(gaussian_img).to(device)
        #saltpepper_img = skimage.util.random_noise(img.cpu(), mode="s&p", amount=0.45)
        #saltpepper_img = torch.from_numpy(saltpepper_img).to(device)
        img_ndarr = (img.cpu()).numpy()

        optimizer.zero_grad()
        #torch.cuda.empty_cache()

        output = model(img.float()).to(device) # feed forward
        #print(output.shape)
        #print(img.shape)
        loss = criterion(output, img)    # calculate loss
        output_ndarr = (output.cpu().detach()).numpy()
        psnr = peak_signal_noise_ratio(img_ndarr,output_ndarr)
        loss.backward()  # calculate new gradients
        optimizer.step()  # update weights
        total_loss += loss.item()  # accumulate loss

    return img, output, total_loss, psnr

def valid():
    with torch.no_grad():
        model.eval().cuda()
        valid_loss = torch.zeros(1)
        for img in valid_loader:
            img = Variable(img).to(device)  # convert to Variable to calculate gradient and move to gpu
            img_ndarr = (img.cpu()).numpy()
            #gaussian_image = skimage.util.random_noise(img.cpu(), mode="gaussian", var=1.6)
            #gaussian_image = torch.from_numpy(gaussian_image).to(device)
            #saltpepper_img = skimage.util.random_noise(img.cpu(), mode="s&p", amount=0.45)
            #saltpepper_img = torch.from_numpy(saltpepper_img).to(device)
            # image, labels = image.to(device), labels.to(device)
            output = model(img.float()).to(device)
            valid_loss += criterion(output, img)  # calculate loss
            output_ndarr = (output.cpu().detach()).numpy()
            psnr = peak_signal_noise_ratio(img_ndarr, output_ndarr)

        return  img, output, valid_loss, psnr

epocharray = []
trainlossarray = []
validlossarray = []
trainsnr = []
validsnr = []
inTotalData = 0


for epoch in range(num_epochs):
    img, output, loss, psnr = train()
    valid_img, valid_output, valid_loss, valid_psnr = valid()

    epocharray.append(epoch)
    trainlossarray.append(loss.item() / len(train_loader))
    validlossarray.append(valid_loss.item() / len(valid_loader))
    trainsnr.append((psnr))
    validsnr.append((valid_psnr))


    print('epoch [{}/{}], loss:{:.4f}, SNR:{}'
        .format(epoch + 1, num_epochs, loss.item()/len(train_loader), psnr))
    print('Validation_loss:{}, SNR: {}'
          .format(valid_loss.item() / len(valid_loader), valid_psnr))

    if epoch % 10 == 0:
        #torch.cuda.empty_cache()
        pic_org = (img)
        pic_pred = (output)
        save_image(pic_org, './denoise_image_org__{}.png'.format(epoch))
        save_image(pic_pred, './denoise_image_pred__{}.png'.format(epoch))

        valid_org = valid_img
        valid_pred = valid_output
        save_image(valid_org, './denoise_image_valid_org__{}.png'.format(epoch))
        save_image(valid_pred, './denoise_image_valid_pred__{}.png'.format(epoch))


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