import io
import torchvision.transforms as transforms
import torch 
import torchvision
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import cv2 
import numpy as np
from skimage.color import rgb2lab, lab2rgb, rgb2gray
from skimage.transform import resize
from skimage.io import imsave
from PIL import Image 
import PIL
from os import getcwd
import matplotlib.pyplot as plt
import base64

PATH = getcwd()
augs_transforms = transforms.Compose([transforms.ToTensor()])

#Model
class CNNColorizer(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size = (3,3), stride= (1,1), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size = (3,3), stride= (1,1), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size = (3,3), stride= (1,1), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size = (3,3), stride= (1,1), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(512, 256, kernel_size = (3,3), stride= (1,1), padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 128, kernel_size = (3,3), stride= (1,1), padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size = (3,3), stride= (1,1), padding=(1,1)),
            nn.ReLU()
        )
        
        self.decoder= nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size = (3,3), stride= (1,1), padding=(1,1)),
            nn.ReLU(),
            nn.Upsample(scale_factor = 2),
            nn.ConvTranspose2d(32, 16, kernel_size = (3,3), stride= (1,1), padding=(1,1)),
            nn.ReLU(),
            nn.Upsample(scale_factor = 2),
            nn.ConvTranspose2d(16, 8, kernel_size = (3,3), stride= (1,1), padding=(1,1)),
            nn.ReLU(),
            nn.Upsample(scale_factor = 2),
            nn.Conv2d(8, 2, kernel_size = (3,3), stride= (1,1), padding=(1,1)),
            nn.Tanh(),
            nn.Upsample(scale_factor = 2)
        )
        
    def forward (self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

device = torch.device('cpu')
model = CNNColorizer().to(device=device)
# MODEL_PATH = PATH + '/model/CNNColorizer-50-flickr-data-kaggle.t7'
MODEL_PATH = PATH + '/model/CNNColorizer-13Apr-transpose-10-flickr-data-kaggle.t7'
# MODEL_PATH = PATH + '/model/CNNColorizer-12Apr-transpose-100-1048pm-kaggle.t7'
checkpoint = torch.load(MODEL_PATH, map_location='cpu')
model.load_state_dict(checkpoint['state_dict'])

def trans_img(input_image):
    img = cv2.imread(PATH + '/'+input_image)
    og = img
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
    img = resize(img, (512,512))
    img = (img - np.min(img))/(np.max(img) - np.min(img))
    lab = rgb2lab(img)
    X = lab[:, :, 0]
    X = X.reshape(X.shape+(1,))
    og = cv2.cvtColor(og, cv2.COLOR_BGR2RGB).astype(np.float32)
    og = resize(og, (512,512))
    # cv2.imwrite(f"{PATH}/static/temp/input.png", og)
    return augs_transforms(X).unsqueeze(0), og
#y = lab[:, :, 1:].transpose((2, 0, 1))
# y = lab[:, :, 1:]
# y = (y/128).astype(np.float32)


def colorize(imagename):
    l, og = trans_img(input_image = imagename)
    l = l.to(device = 'cpu')
    op = model(l)
    op = op*128
    op = op[0].permute(1,2,0).detach().numpy()

    result = np.zeros((512,512,3))
    result[:, :, 0:1] = l.cpu()[0].permute(1,2,0).numpy()
    result[:, :, 1:2 ] = op[:,:,0:1]
    result[:, :, 2:3 ] = op[:,:,1:2]
    colorimage = lab2rgb(result)
    colorimage = 255*(colorimage - np.min(colorimage))/(np.max(colorimage) - np.min(colorimage))
    colorimage = colorimage.astype(np.uint8)
    colorimage = cv2.cvtColor(colorimage, cv2.COLOR_BGR2RGB)

    # cv2.imwrite(f"{PATH}/static/temp/output.png", colorimage)
    _, input_img = cv2.imencode('.jpg', og)  
    input_bytes = input_img.tobytes()
    input_b64 = base64.b64encode(input_bytes)
    input_b64 = input_b64.decode('utf-8')

    _, output_img = cv2.imencode('.jpg', colorimage)  
    output_bytes = output_img.tobytes()
    output_b64 = base64.b64encode(output_bytes)
    output_b64 = output_b64.decode('utf-8')

    result = {
        "input":f'data:image/jpg;base64,{input_b64}',
        "output":f'data:image/jpg;base64,{output_b64}'
    }

    return result
# imsave('colored.jpg', lab2rgb(result))

# colorize('bandw.jpg')

# plt.imshow(colorimage)
# plt.show()