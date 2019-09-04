# -*- coding: utf-8 -*-
"""
Utility functions
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

from PIL import Image, ImageOps

import os
import zipfile
import math
import re
import struct
import time
from datetime import datetime
from collections import namedtuple
from glob import glob


###############################
# Dataset
###############################

class CustomImageFolder(dset.ImageFolder):
    """
     Extend ImageFolder dataset to include image path in the return tuple.

     Attributes:
        classes (list)     : List of the class names (i.e. folder names).
        class_to_idx (dict): Dict with items {class_name: class_index}.
        imgs (list)        : List of (image path, class_index) tuples
    """

    def __getitem__(self, index):
        # This is what ImageFolder normally returns 
        sample, target = super().__getitem__(index)
        # Get image path
        path = self.imgs[index][0]
        return sample, target, path
    
class ChunkSampler(sampler.Sampler):
    """Samples elements sequentially from some offset. 
    
    Arguments:
        num_samples: Number of desired datapoints
        start:       Offset where we should start selecting from
        shuffle:     Set to True to have the data reshuffled at every epoch
    """
    def __init__(self, num_samples, start=0, shuffle=False):
        self.num_samples = num_samples
        self.start       = start
        self.shuffle     = shuffle

    def __iter__(self):
        if self.shuffle:
            indices = range(self.start, self.start + self.num_samples)
            return (indices[i] for i in torch.randperm(len(indices)))
        else:
            return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples

# Pytorch ImageFolder's default PIL loader returns "RGB" image.
# This loader returns the image without any conversion.
def pil_vanilla_loader(path):
    return Image.open(path)
    
# Return greyscale ("L") image.
def pil_l_loader(path):
    # Open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('L')

# Wrapper for Pytorch datasets
def load_data(dataset, path, transform, batch_size=128, num_workers=0, debug=False):
    if (dataset == "mnist"):
        train_data = path + './MNIST_data'
        val_data   = path + './MNIST_data'
        
        colour_img = False
        
        # MNIST dataset: 60,000 training, 10,000 test images.
        # We'll take NUM_VAL of the training examples and place them into a validation dataset.
        NUM_TRAIN = 55000
        NUM_VAL   = 5000
        
        # Training set
        mnist_train = dset.MNIST(root=train_data,
                                 train=True, download=True,
                                 transform=transform)
        train_loader = DataLoader(mnist_train,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  sampler=ChunkSampler(NUM_TRAIN, 0),
                                  drop_last=True)
        # Validation set
        mnist_val = dset.MNIST(root=val_data,
                               train=True, download=True,
                               transform=transform)
        val_loader = DataLoader(mnist_val,
                                batch_size=batch_size,
                                num_workers=num_workers,
                                sampler=ChunkSampler(NUM_VAL, start=NUM_TRAIN),
                                drop_last=True)
    elif (dataset == "cifar"):
        colour_img   = True
        archive_path = path + 'CIFAR.zip'
        extract_path = './'
        train_data   = extract_path + 'CIFAR'
        val_data     = extract_path + 'CIFAR'
        zip_ref      = zipfile.ZipFile(archive_path, 'r')
        zip_ref.extractall(extract_path)
        zip_ref.close()
        
        # Training set
        train_dataset = dset.CIFAR10(root=train_data,
                                     train=True, 
                                     download=True,
                                     transform=transform)
        train_loader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=(not debug),
                                  num_workers=num_workers,
                                  drop_last=True)
    
        # Validation set
        val_dataset = dset.CIFAR10(root=val_data,
                                   train=False, 
                                   download=True,
                                   transform=transform)
        val_loader  = DataLoader(val_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 num_workers=num_workers,
                                 drop_last=True)
    elif (dataset == "kkanji"):
        # kkanji2 dataset has 3818 characters and 140,384 images.
        colour_img   = False
        archive_path = path + 'kkanji2_32x32_split.zip'
        extract_path = './'
        train_data   = extract_path + 'kkanji2_32x32_split/train'
        val_data     = extract_path + 'kkanji2_32x32_split/validation'
        zip_ref      = zipfile.ZipFile(archive_path, 'r')
        zip_ref.extractall(extract_path)
        zip_ref.close()

        # Training set
        train_dataset = dset.ImageFolder(root=train_data,
                                         transform=transform,
                                         loader=pil_l_loader)
        train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=(not debug), num_workers=num_workers)
    
        # Validation set
        val_dataset = dset.ImageFolder(root=val_data,
                                       transform=transform,
                                       loader=pil_l_loader)
        val_loader  = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    else:
        colour_img   = False
        archive_path = path + dataset
        train_data   = archive_path
        if dataset.endswith('.zip'):
            extract_path = './'
            train_data   = extract_path + dataset
            zip_ref      = zipfile.ZipFile(archive_path, 'r')
            zip_ref.extractall(extract_path)
            zip_ref.close()
    
        train_dataset = dset.ImageFolder(root=train_data,
                                         transform=transform,
                                         loader=pil_l_loader)
        train_loader  = DataLoader(train_dataset, batch_size=batch_size, shuffle=(not debug), num_workers=num_workers)
    
        val_loader = None
        
    return train_loader, val_loader, train_dataset, val_dataset

###############################
# Enhanced dictionary class    
###############################

# __missing__() gets called when a non-present dict key is used. This way we
# don't need to initialise the dict before using.
# See https://docs.python.org/3.7/library/stdtypes.html#d[key]

# collections module already has Counter() which support this usage and more.
class Counter(dict):
    def __missing__(self, key):
        return 0

class CounterArray(dict):
    def __init__(self, size=1):
        super().__init__()
        self.size = size
        
    def __missing__(self, key):
        self[key] = [0] * self.size
        return self[key]

###############################
# Pytorch
###############################

def count_params(model):
    """Count the number of parameters in the current model graph """
    param_count = np.sum([np.prod(p.size()) for p in model.parameters()])
    return param_count

###############################
# Averaging functions
###############################
# The loss on individual batches is quite noisy, so an exponentially decayed moving 
# average to smooth out individual fluctuations in each batch makes general trends
# easier to see.
# At validation time you usually want to find the average loss over the full dataset,
# so a running average (cumulative moving average) is more appropriate.
    
class AverageBase(object):
    
    def __init__(self, value=0):
        self.value = float(value) if value is not None else None
       
    def __str__(self):
        return str(round(self.value, 4))
    
    def __repr__(self):
        return self.value
    
    def __format__(self, fmt):
        return self.value.__format__(fmt)
    
    def __float__(self):
        return self.value
    
class RunningAverage(AverageBase):
    """
    Keeps track of a cumulative moving average (CMA).
    """
    
    def __init__(self, value=0, count=0):
        super(RunningAverage, self).__init__(value)
        self.count = count
        
    def update(self, value):
        self.value  = (self.value * self.count + float(value))
        self.count += 1
        self.value /= self.count
        return self.value

class DecayAverage(AverageBase):
    """
    An exponentially decaying moving average (EMA).
    """
    
    def __init__(self, alpha=0.99):
        super(DecayAverage, self).__init__(None)
        self.alpha = alpha
        
    def update(self, value):
        if self.value is None:
            self.value = float(value)
        else:
            self.value = self.alpha * self.value + (1 - self.alpha) * float(value)
        return self.value
        
###############################
# Image processing
###############################

class Flatten(nn.Module):
    """
    Receives an input of shape (N, C, H, W) and "flattens" the C, H and W dimensions
    into a single vector, or alternatively flatten just the H and W dimensions. The
    batch size dimension (N) is always kept.
    """
    def __init__(self, keep_channel=False):
        super().__init__()
        self.keep_channel = keep_channel
        
    def forward(self, x):
        N, C, H, W = x.size() # N = minibatch size, C = num of channels, (H)eight, (W)idth
        if (self.keep_channel):
            return x.view(N, C, -1)
        else:
            return x.view(N, -1)
    
class Unflatten(nn.Module):
    """
    An Unflatten module receives an input of shape (N, C*H*W) and reshapes it
    to produce an output of shape (N, C, H, W).
    """
    def __init__(self, N, C, H, W):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W
        
    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)

###############################
# Show image/plots
###############################

# Input should be a tensor of size (N, C, H, W) where N is the total
# number of images to display, and C is the colour channel.
def show_colour_images(images):
    plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    
    # pyplot expects colour channel in last dimension
    images = np.transpose(images, (0, 2, 3, 1))
    
    # Grid of sqrtn-by-sqrtn images
    sqrtn  = int(np.ceil(np.sqrt(images.shape[0])))  
    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs  = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img)
    plt.show()
    
    return fig

def show_gray_images(images):
    plt.rcParams['figure.figsize'] = (10.0, 8.0) # set default size of plots
    plt.rcParams['image.interpolation'] = 'nearest'
    plt.rcParams['image.cmap'] = 'gray'
    
#    img_h   = images.shape[1]
#    img_w   = images.shape[2]
    images  = np.reshape(images, [images.shape[0], -1])
    sqrtn   = int(np.ceil(np.sqrt(images.shape[0])))
    sqrtimg = int(np.ceil(np.sqrt(images.shape[1])))
    
    fig = plt.figure(figsize=(sqrtn, sqrtn))
    gs  = gridspec.GridSpec(sqrtn, sqrtn)
    gs.update(wspace=0.05, hspace=0.05)

    for i, img in enumerate(images):
        ax = plt.subplot(gs[i])
        plt.axis('off')
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_aspect('equal')
        plt.imshow(img.reshape([sqrtimg, sqrtimg]))
 #       plt.imshow(img.reshape([img_h, img_w]))
    plt.show()
    
    return fig

def show_images(images, colour):
    if colour:
        return show_colour_images(images)
    else:
        return show_gray_images(images)
    
def preprocess_img(x):
    return 2 * x - 1.0

def deprocess_img(x):
    return (x + 1.0) / 2.0

def rel_error(x,y):
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def plot_loss(grid, starting_epoch, epoch, train_losses, val_losses):
    if isinstance(train_losses, dict):
        iterator = train_losses.items()
    else:
        iterator = train_losses._asdict().items()
        
    for i, (loss_type, loss_data) in enumerate(iterator):
        with grid.output_to(0, i):
            grid.clear_cell()
            plt.figure(figsize=(10,6))
            plot_epochs = range(starting_epoch, epoch + 1)
            plt.plot(plot_epochs, loss_data, '-o', label='Training')
            plt.legend()
            plt.title('Training ' + loss_type + ' loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.xticks(plot_epochs)
            plt.show()
            
    if isinstance(val_losses, dict):
        iterator = val_losses.items()
    else:
        iterator = val_losses._asdict().items()
        
    for i, (loss_type, loss_data) in enumerate(iterator):
        with grid.output_to(1, i):
            grid.clear_cell()
            plt.figure(figsize=(10,6))
            plot_epochs = range(starting_epoch, epoch + 1)
            plt.plot(plot_epochs, loss_data, '-o', label='Validation')
            plt.legend()
            plt.title('Validation ' + loss_type + ' loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.xticks(plot_epochs)
            plt.show()

###############################
# Function hooks
###############################

# Callback functions for Pytorch hooks to aid debugging

def printnorm(self, fwd_input, fwd_output):
    # Input is a tuple of packed inputs
    # Output is a Tensor. output.data is the Tensor we are interested
    if torch.isnan(fwd_input[0]).any() or torch.isnan(fwd_output[0]).any():
        print('\nForward inside class: ' + self.__class__.__name__)
        print('input: ', fwd_input)
        print('output: ', fwd_output)
        print('output norm:', fwd_output.data.norm())
        print('input size:', fwd_input[0].size())
        print('output size:', fwd_output.data.size())
    assert not torch.isnan(fwd_input[0]).any(), "fwd_input[0] has NaN"
    assert not torch.isnan(fwd_output[0]).any(), "fwd_output[0] has NaN"

def printgrad(self, grad_input, grad_output):
    for i, g in enumerate(grad_output[0]):
        if torch.isnan(g).any():
            print('\nBackprop inside class: ' + self.__class__.__name__)
            print('grad_output: ', g)
            print('grad_input: ', grad_input[0][i])
            print('grad_input norm:', grad_input[0][i].norm())
            debug_dict[i] = {'grad in'  : grad_input[0][i],
                             'grad out' : g}
            assert not torch.isnan(g).any(), "grad_output has NaN"
            assert not torch.isnan(grad_input[0][i]).any(), "grad_input has NaN"
    #assert not torch.isnan(grad_output[0]).any(), "grad_output has NaN"
    #assert not torch.isnan(grad_input[0]).any(), "grad_input has NaN"

def check_backprop(name):
    def hook(grad):
        for i, g in enumerate(grad):
            if torch.isnan(g).any():
                print('\nGradient w.r.t. ' + name + ":")
                print(g.size())
                print('grad: ', g)
                print('grad norm:', g.norm())
                debug_dict['{}_{}'.format(name, i)] = {'grad' : g}
                assert not torch.isnan(g).any(), "gradient has NaN"
    return hook

###############################
# PNG to IDX format
###############################

# Modified based on https://www.josephcatrambone.com/?page_id=247
def png2idx(in_path, out_file, scale_size=(64, 64)):
    imglist = glob(in_path, recursive=True)
    fout    = open(out_file, 'wb')

	# Write header
    #  - Pack two bytes with zero zero
    fout.write(struct.pack(">bb", 0, 0))
    #  - Output format
    #     0x08 : unsigned byte
    #     0x09 : signed byte
    #     0x0B : 2-byte short
    #     0x0C : 4-byte int
    #     0x0D : 4-byte float
    #     0x0E : 8-byte double
    fout.write(struct.pack(">b", 0x0D))
    # Storing a matrix of values, 2 dimensions.
    fout.write(struct.pack(">b", 0x2))
    # Dimension one size (number of images)
    fout.write(struct.pack(">i", len(imglist)))
    # Dimension two size (image data)
    fout.write(struct.pack(">i", scale_size[0]*scale_size[1]))
    
    for imgname in imglist:
        if not imgname.endswith('.png'):
            continue
            
        try:
            img = Image.open(imgname)
            img = img.resize(scale_size)
        except:
            print("Error occured trying to open: {}".format(imgname))
            for _ in range(scale_size[0]*scale_size[1]):
                fout.write(struct.pack(">f", 0x0))
            continue
    
        if img.mode == 'RGB':
            print("Color images not supported. Converting to grey.")
            img = img.convert('L')
        if img.mode == 'L':
            data = img.getdata()
            for d in data:
                fout.write(struct.pack(">f", d))
        else:
            print("Image format not recognized.")
            for _ in range(scale_size[0]*scale_size[1]):
                fout.write(struct.pack(">f", 0x0))
            continue

    fout.close()