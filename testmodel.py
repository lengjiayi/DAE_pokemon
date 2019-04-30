import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import csv

from PIL import Image

class ATE(nn.Module):
	def __init__(self):
		super(ATE,self).__init__()
		self.encoder = nn.Sequential(
			nn.Linear(400*167,400),
#			nn.ReLU(True),
			nn.Linear(400,128),
#			nn.ReLU(True),
			nn.Linear(128,64),
#			nn.ReLU(True),
			nn.Linear(64,12),
#			nn.ReLU(True),
			nn.Linear(12,5),
			)
		self.decoder = nn.Sequential(
			nn.Linear(5,12),
#			nn.ReLU(True),
			nn.Linear(12,64),
#			nn.ReLU(True),
			nn.Linear(64,128),
#			nn.ReLU(True),
			nn.Linear(128,400),
#			nn.ReLU(True),
			nn.Linear(400,400*167),
			nn.Sigmoid(),
			)
	def forward(self,x):
		x=self.encoder(x)
		x=self.decoder(x)
		return x


def getdata():
	sample = []
	text = open('pixel_color.txt','r',encoding='utf-8')
	row = csv.reader(text, delimiter=' ')
	for r in row:
		pict = []
		for pix in r:
			index = int(pix)
			tmp = np.zeros(167)
			tmp[index]=1
			pict.append(tmp)
		pict = np.array(pict)		
		sample.append(pict.reshape(400*167))
	sample = np.array(sample)
	return sample


def getMap():
	cmap = []
	text = open('colormap.txt','r',encoding='utf-8')
	row = csv.reader(text, delimiter=' ')
	i = 0
	for r in row:
		cmap.append([])		
		for c in r:
			cmap[i].append(int(c))
		i += 1
	return cmap


def getHyp(X):
	return X.argmax(axis=0)


def getImg(code):
	img=[]
	for x in code:
		t = getHyp(x)
		t = min(t, len(cmap)-1)
		img.append(cmap[t])
	return np.array(img).reshape(20,20,3)


def vitest():
	for i in range(0,5):
		norm = ate(testset[0].float())[i]
		norm = norm.detach().numpy()
		norm = norm.reshape(400,167)
		testimg = getImg(norm)
		rawimg = getImg(testset[0][i].numpy().reshape(400,167))
		plt.subplot(5,2,i*2+1)
		plt.xticks([])
		plt.yticks([])
		plt.imshow(rawimg)
		plt.subplot(5,2,i*2+2)
		plt.xticks([])
		plt.yticks([])
		plt.imshow(testimg)
	plt.show()

ate = ATE()
ate.load_state_dict(torch.load('autoencoder.pth'))

sample = getdata()
cmap = getMap()

testset = list(DataLoader(sample, batch_size=5, shuffle=True))
vitest()
'''
codes=[]
for x in testset:
	codes.append(ate.encoder(x.float())[0].detach().numpy())
codes = np.array(codes)

maxfield=codes.max(axis=0)
minfield=codes.min(axis=0)

for i in range(0,11):
	for j in range(0,11):
		tmp=codes[k]
		tmp[channel1]=0.99/10*i
		tmp[channel1]=tmp[channel1]*(maxfield-minfield)[channel1]+minfield[channel1]
		tmp[channel2]=0.99/10*j	
		tmp[channel2]=tmp[channel2]*(maxfield-minfield)[channel2]+minfield[channel2]
		norm = ate.decoder(torch.from_numpy(tmp).float())
		norm = norm.detach().numpy()
		norm = norm.reshape(400,167)
		testimg = getImg(norm)
		plt.subplot(11,11,i*11+j+1)
		plt.xticks([])
		plt.yticks([])
		plt.imshow(testimg)
plt.subplots_adjust(wspace=0,hspace=0,left=None,right=None,bottom=None,top=None)
plt.show()
'''
'''
pre=codes[106]
now=codes[102]

inter=(now-pre)/9
for i in range(0,10):
	tmp=pre
	tmp = tmp+inter*i
	norm = ate.decoder(torch.from_numpy(tmp).float())
	norm = norm.detach().numpy()
	norm = norm.reshape(400,167)
	testimg = getImg(norm)
	img = Image.fromarray(np.uint8(testimg)).resize((40,40))
	img.save('slice'+str(i+11)+'.png')
#	plt.subplot(1,10,i+1)
#	plt.xticks([])
#	plt.yticks([])
#	plt.imshow(testimg)
#plt.subplots_adjust(wspace=0,hspace=0,left=None,right=None,bottom=None,top=None)
#plt.show()
'''

for i in range(100,109):
	tmp = sample[i]
	tmp = tmp.reshape(400,167)
	img = getImg(tmp)
	img = Image.fromarray(np.uint8(img)).resize((40,40))
	img.save(str(i)+'.png')