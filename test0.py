import argparse
import torch
import utils
from PIL import Image
from tqdm import tqdm
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms
from torchvision.transforms import ToTensor,ToPILImage
from PIL import Image

import numpy as np

from net.model import net_residual as Net
import time


parser = argparse.ArgumentParser(description="PyTorch DeepDehazing")
parser.add_argument("--checkpoint", type=str, default="../ITS-data/489.pth",help="path to load model checkpoint")
parser.add_argument("--test", type=str, default="../ITS-data/test/data/", help="path to load test images")

opt = parser.parse_args()

net = Net()
print(opt)
net.load_state_dict(torch.load(opt.checkpoint)['state_dict'])
net.eval()
net = nn.DataParallel(net, device_ids=[0]).cuda()

images = utils.load_all_image(opt.test)

time_sum = 0
ssims = []
psnrs = []
psnrs2 = []
str_label = "../ITS-data/test/label/"

for im_path in tqdm(images):
    filename = im_path.split('/')[-1]
    start = time.time()
    im2_path = str_label + filename.split('_')[0] + '.png'
    im2 = Image.open(im2_path)
    im = Image.open(im_path)
    h, w = im.size
    im = ToTensor()(im)
    im = im.unsqueeze(0)
    im = Variable(im, volatile=True)
    im = im.cuda()
    im = net(im)
    im = torch.clamp(im, 0., 1.)
    im = im.cpu()
    im = im.data[0]
    im = ToPILImage()(im)
    end = time.time()
    t = end-start
    time_sum += t
    im.save('outdoor/%s' % filename)
