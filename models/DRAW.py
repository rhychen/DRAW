# -*- coding: utf-8 -*-
"""
PyTorch implementation of DRAW

[1] Gregor, K., Danihelka, I., Graves, A., Rezende, D. & Wierstra, D.. 2015. DRAW: A Recurrent Neural Network For Image Generation. PMLR 37:1462-1471
[2] Danilo J. Rezende, Shakir Mohamed, Ivo Danihelka, Karol Gregor, and Daan Wierstra. 2016. One-shot generalization in deep generative models. ICML'16
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import sampler
import torchvision
import torchvision.transforms as T
import torchvision.datasets as dset
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os          
import time
from utility import *


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# Output directory
output_dir = 'DRAW_out/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    
chkpt_dir = 'DRAW_chkpt/'
if not os.path.exists(chkpt_dir):
    os.makedirs(chkpt_dir)
    
###############################
# Hyperparameters
###############################

# Dataset
batch_size = 128

# Number of recurrent steps (glimpses)
recur_length = 20
    
# Number of units in each hidden layer (excl. latent code layer)
h_len = 256

# Length of latent code (number of units in latent code layer)
z_len = 100

# N-by-N grid of attention filters
N = 10

# Training
num_epochs = 100
lr         = 1e-3

###############################
# Load data
###############################

# MNIST dataset: 60,000 training, 10,000 test images.
# We'll take NUM_VAL of the training examples and place them into a validation dataset.
NUM_TRAIN  = 55000
NUM_VAL    = 5000

# Training set
mnist_train = dset.MNIST('C:/datasets/MNIST',
                         train=True, download=True,
                         transform=T.ToTensor())
loader_train = DataLoader(mnist_train, batch_size=batch_size,
                          sampler=ChunkSampler(NUM_TRAIN, 0))
# Validation set
mnist_val = dset.MNIST('C:/datasets/MNIST',
                       train=True, download=True,
                       transform=T.ToTensor())
loader_val = DataLoader(mnist_val, batch_size=batch_size,
                        sampler=ChunkSampler(NUM_VAL, start=NUM_TRAIN))
# Test set
mnist_test = dset.MNIST('C:/datasets/MNIST',
                       train=False, download=True,
                       transform=T.ToTensor())
loader_test = DataLoader(mnist_test, batch_size=batch_size, shuffle=True)

# Input dimensions
train_iter     = iter(loader_train)
images, labels = train_iter.next()
_, C, H, W     = images.size()
img_flat_dim   = H * W          # Flattened image dimension

###############################
# Model
###############################

class draw(nn.Module):
    def __init__(self, recur_length, img_flat_dim, H, W, h_len, z_len, N, use_attn=True):
        super().__init__()
        self.recur_length = recur_length
        self.img_flat_dim = img_flat_dim
        self.img_H        = H
        self.img_W        = W
        self.h_len        = h_len
        self.z_len        = z_len
        self.N            = N
        self.use_attn     = use_attn
        self.attn_params  = 5       # Number of parameters for attention mechanism

        if (use_attn):          
            self.encoder     = nn.LSTMCell(2 * N * N + h_len, h_len)
            self.fc_wr_patch = nn.Linear(h_len, N * N)
        else:
            self.encoder     = nn.LSTMCell(2 * img_flat_dim + h_len, h_len)
            self.fc_wr_patch = nn.Linear(h_len, img_flat_dim)
        self.decoder    = nn.LSTMCell(z_len, h_len)
        self.fc_mean    = nn.Linear(h_len, z_len)
        self.fc_var     = nn.Linear(h_len, z_len)
        self.fc_wr      = nn.Linear(h_len, img_flat_dim)
        self.fc_attn_rd = nn.Linear(h_len, self.attn_params)
        self.fc_attn_wr = nn.Linear(h_len, self.attn_params)

    def reparameterize(self, mean, logvar):
        sd  = torch.exp(0.5 * logvar)   # Standard deviation
        # Latent distribution uses Gaussian
        eps = torch.randn_like(sd)      
        z   = eps.mul(sd).add(mean)
        return z
    
    def get_attn_filters(self, params, epsilon=1e-9, debug=False):
        gx_hat, gy_hat, log_var, log_delta_hat, log_gamma = params.split(1,1)
        
        # Debug aid
        if (debug):
            gx_hat = torch.zeros(params.size(0), 1)
            gy_hat = torch.zeros(params.size(0), 1)
            log_var = torch.zeros(params.size(0), 1)
            log_delta_hat = torch.zeros(params.size(0), 1)
            log_gamma = torch.zeros(params.size(0), 1)
        
        # gx, gy, and stride are scaled to ensure the initial patch covers the
        # entire input image. Variance, stride and intensity are emitted in the
        # log-scale to ensure positivity.
        gx        = (gx_hat + 1) * (self.img_W + 1) / 2
        gy        = (gy_hat + 1) * (self.img_H + 1) / 2
        var       = torch.exp(log_var)
        stride    = (max(self.img_W, self.img_H) - 1) * torch.exp(log_delta_hat) / (self.N - 1)
        intensity = torch.exp(log_gamma)
        
        # Filter bank (filters' centre coordinates)
        centres_x = gx + (torch.arange(1, self.N + 1) - self.N / 2 - 0.5).float() * stride
        centres_y = gy + (torch.arange(1, self.N + 1) - self.N / 2 - 0.5).float() * stride
        
        # Filter matrices
        # Reshape filter bank into (batch_size, N, 1) tensor and use broadcasting to get
        # batch_size-by-N-by-H and batch_size-N-by-W (unnormalised) matrices
        raw_filter_x = torch.exp(-(torch.arange(self.img_H, dtype=torch.float) - centres_x.view(-1, self.N, 1))**2 / (2 * var).view(-1, 1, 1))
        raw_filter_y = torch.exp(-(torch.arange(self.img_W, dtype=torch.float) - centres_y.view(-1, self.N, 1))**2 / (2 * var).view(-1, 1, 1))

        # Normalise
        # Dimension 0 is batch size, dim 1 is N. Sum along dimension 2 (image H or W) so
        # the normalising constant is of dim [batch_size, N, 1]
        normalising_const_x = raw_filter_x.sum(2, keepdim=True)
        normalising_const_y = raw_filter_y.sum(2, keepdim=True)
        # Add a small epsilon to prevent division by zero
        filter_x = raw_filter_x / (normalising_const_x + epsilon)
        filter_y = raw_filter_y / (normalising_const_y + epsilon)
        
        # Debug aid
        if (debug):
            print("raw_filter_x size: ", raw_filter_x.size())
            filter_x_numpy = filter_x.transpose(1, 2).detach().to('cpu').numpy()
            fig = show_images(filter_x_numpy[0:16])
            plt.close(fig)
            filter_y_numpy = filter_y.detach().to('cpu').numpy()
            fig = show_images(filter_y_numpy[0:16])
            plt.close(fig)
        
        return intensity, filter_x, filter_y
        
    def read(self, in_img_flat, err_img_flat, h_dec, debug=False):
        
        if (self.use_attn):          
            # Linear transformation of decoder output to get attention parameters
            params = self.fc_attn_rd(h_dec)

            # Unflatten images for matrix multiplication
            in_img  = in_img_flat.view(-1, self.img_H, self.img_W)
            err_img = err_img_flat.view(-1, self.img_H, self.img_W)
            
            # REVISIT: Insteady of applying the same intensity to all, try having
            # REVISIT: an intensity parameter for each of the N-by-N points
            intensity, filter_x, filter_y = self.get_attn_filters(params)
            in_patch  = intensity.view(-1, 1, 1) * torch.matmul(torch.matmul(filter_y, in_img.transpose(1, 2)), filter_x.transpose(1, 2))
            err_patch = intensity.view(-1, 1, 1) * torch.matmul(torch.matmul(filter_y, err_img.transpose(1, 2)), filter_x.transpose(1, 2))
            rd_patch  = torch.cat((in_patch.view(-1, self.N * self.N), err_patch.view(-1, self.N * self.N)), dim=1)
            
            # Debug aid
            if (debug):
                print("filter_x size: ", filter_x.size())
                print("in_patch size: ", in_patch.size())
                in_img_numpy = in_img.detach().to('cpu').numpy()
                fig = show_images(in_img_numpy[0:16])
                plt.close(fig)
                in_patch_numpy = in_patch.detach().to('cpu').numpy()
                fig = show_images(in_patch_numpy[0:16])
                plt.close(fig)
        else:
            # No attention
            rd_patch = torch.cat((in_img_flat, err_img_flat), dim=1)
                
        return rd_patch
    
    def write(self, h_dec):
        if (self.use_attn):          
            params        = self.fc_attn_wr(h_dec)
            wr_patch_flat = self.fc_wr_patch(h_dec)
            
            intensity, filter_x, filter_y = self.get_attn_filters(params)
            # Unflatten for matrix multiplication
            wr_patch_raw  = wr_patch_flat.view(-1, self.N, self.N)
            
            wr_patch = intensity.view(-1, 1, 1) * torch.matmul(torch.matmul(filter_y.transpose(1, 2), wr_patch_raw), filter_x)
            # Flatten
            wr_patch_flat = wr_patch.view(-1, self.img_flat_dim)
        else:
            wr_patch_flat = self.fc_wr_patch(h_dec)
        
        return wr_patch_flat
    
    def forward(self, mb_size, in_image):
        # Initialisation
        h_enc   = torch.zeros(mb_size, self.h_len, requires_grad=True) # Hidden state
        c_enc   = torch.zeros(mb_size, self.h_len, requires_grad=True) # Cell state
        h_dec   = torch.zeros(mb_size, self.h_len, requires_grad=True)
        c_dec   = torch.zeros(mb_size, self.h_len, requires_grad=True)
        canvas  = [torch.zeros_like(in_image, requires_grad=True)]
        means   = [0] * self.recur_length
        logvars = [0] * self.recur_length
        
        # REVISIT: [2] claims a single LSTM shared by encoder and decoder is enough (section 3.2.5)
        for t in range(self.recur_length):
            err_image = in_image - torch.sigmoid(canvas[t])
            r         = self.read(in_image, err_image, h_dec)
            
            h_enc, c_enc = self.encoder(torch.cat((r, h_dec), dim=1), (h_enc, c_enc))
            means[t]     = self.fc_mean(h_enc)
            logvars[t]   = self.fc_var(h_enc)
            z            = self.reparameterize(means[t], logvars[t])
            h_dec, c_dec = self.decoder(z, (h_dec, c_dec))

            canvas.append(canvas[t] + self.write(h_dec))
            
        # Black and white reconstruction
        recon = torch.sigmoid(canvas[-1])
        
        return recon, means, logvars
    
    def generate_img(self, mb_size):     
        h_dec  = torch.zeros(mb_size, self.h_len)
        c_dec  = torch.zeros(mb_size, self.h_len)
        canvas = torch.zeros(mb_size, self.img_flat_dim)

        for t in range(self.recur_length):
            z            = torch.randn(mb_size, self.z_len).to(device)
            h_dec, c_dec = self.decoder(z, (h_dec, c_dec))
            canvas      += self.write(h_dec)

        gen_img = torch.sigmoid(canvas)
        return gen_img

###############################
# Loss function
###############################

def loss_fn(original, recon, means, logvars):
    # Reconstruction loss is the negative log probability, i.e. Bernoulli cross-entropy
    # for black and white images
    recon_loss = F.binary_cross_entropy(recon, original, reduction='sum')
    
    KLD = 0
    for t in range(recur_length):
        # Latent loss is the negative KL divergence
        KLD -= 0.5 * torch.sum(1 + logvars[t] - means[t] ** 2 - logvars[t].exp())
    
    return recon_loss + KLD

###############################
# Main
###############################

model     = draw(recur_length, img_flat_dim, H, W, h_len, z_len, N).to(device)
optimiser = optim.Adam(model.parameters(), lr=lr)

def train(epoch):
    start_time = time.time()
    model.train()
    loss_train = 0
    for batch, (x, _) in enumerate(loader_train):       
        # Reshape (flatten) input images
        x_flat = x.view(x.size(0), -1).to(device)
        
        with torch.autograd.detect_anomaly():
            recon, means, logvars = model(len(x_flat), x_flat)
            loss        = loss_fn(x_flat, recon, means, logvars)
            loss_train += loss.item()
              
            loss.backward()
            
        optimiser.step()
        optimiser.zero_grad()

    epoch_time = time.time() - start_time
    print("Time Taken for Epoch %d: %.2fs" %(epoch, epoch_time))
    
    if epoch % 10 == 0:
        print("Epoch {} reconstruction:".format(epoch))
        imgs_numpy = recon.detach().to('cpu').numpy()
        fig = show_images(imgs_numpy[0:16])
        plt.close(fig)
    
    print('Epoch {} avg. training loss: {:.3f}'.format(epoch, loss_train / len(loader_train.dataset)))

def validation(epoch):
    model.eval()
    loss_val = 0
    with torch.no_grad():
        for batch, (x, _) in enumerate(loader_val):
            # Reshape (flatten) input images
            x_flat = x.view(x.size(0), -1).to(device)

            recon, mean, logvar = model(len(x_flat), x_flat)
            loss      = loss_fn(x_flat, recon, mean, logvar)
            loss_val += loss.item()
            
    print('Epoch {} validation loss: {:.3f}'.format(epoch, loss_val / len(loader_val.dataset)))

for epoch in range(num_epochs):
    print('Epoch {}'.format(epoch))
    train(epoch)
    validation(epoch)
    if epoch % 2 == 0:
        with torch.no_grad():
            print("Epoch {} generation:".format(epoch))
            sample = model.generate_img(32)
            imgs_numpy = sample.to('cpu').numpy()
            fig = show_images(imgs_numpy[0:16])
            # Save image to disk
            fig.savefig('{}/{}.png'.format(output_dir, str(epoch).zfill(3)), bbox_inches='tight')
            plt.close(fig)

        # Save checkpoints
        torch.save({
            'model'      : model.state_dict(),
            'optimiser'  : optimiser.state_dict(),
            'hyperparams': {'recur_length': 20,
                            'h_len'       : 256,
                            'z_len'       : 100,
                            'N'           : 10,
                            'num_epochs'  : 100}
        }, '{}/epoch_{}.pth'.format(chkpt_dir, epoch))

