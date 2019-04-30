import torch
import torchvision

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import numpy as np
import csv



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
		rawimg = getImg(sample[i].reshape(400,167))
		plt.subplot(5,2,i*2+1)
		plt.imshow(rawimg)
		plt.subplot(5,2,i*2+2)
		plt.imshow(testimg)
	plt.show()



def train(epoch_num):
	ate.train()
	for epoch in range(0,epoch_num):
		for data in dataloader:
			output = ate(data.float())
			loss = criterion(output,data.float())
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		print('epoch:{}/{}: BCEloss:{:.8f}'.format(epoch,
		epoch_num,loss.item()))
		if epoch%20==19:
			vitest()


criterion = nn.BCELoss()

learning_rate = 0.001
epoch_num = 100
batch_size = 100

plt.figure(1)

ate = ATE()
ate.load_state_dict(torch.load('autoencoder.pth'))
optimizer = optim.Adam(ate.parameters(),lr=learning_rate)
optimizer.load_state_dict(torch.load('opt.pth'))

sample = getdata()
cmap = getMap()
dataloader = DataLoader(sample, batch_size=batch_size, shuffle=True)
testset = list(DataLoader(sample, batch_size=5, shuffle=False))
train(epoch_num)
vitest()
torch.save(ate.state_dict(),'./autoencoder.pth')
torch.save(optimizer.state_dict(),'./opt.pth')
