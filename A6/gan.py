from __future__ import print_function
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.autograd import Variable
from torch.nn import functional as F
from torchvision import datasets, transforms
from torchvision.utils import save_image
from collections import OrderedDict
NOISE_DIM = 96

def hello_gan():
    print("Hello from gan.py!")


def sample_noise(batch_size, noise_dim, dtype=torch.float, device='cpu'):
  """
  Generate a PyTorch Tensor of uniform random noise.

  Input:
  - batch_size: Integer giving the batch size of noise to generate.
  - noise_dim: Integer giving the dimension of noise to generate.
  
  Output:
  - A PyTorch Tensor of shape (batch_size, noise_dim) containing uniform
    random noise in the range (-1, 1).
  """
  noise = None
  ##############################################################################
  # TODO: Implement sample_noise.                                              #
  ##############################################################################
  noise = 2 * torch.rand(batch_size, noise_dim, dtype=dtype, device=device) - 1
  return noise



def discriminator():
  """
  Build and return a PyTorch nn.Sequential model implementing the architecture in the notebook.
  """
  model = None
  ############################################################################
  # TODO: Implement discriminator.                                           #
  ############################################################################
  model = nn.Sequential(OrderedDict([
    ('Linear1', nn.Linear(784, 256)),
    ('LeakyReLU1', nn.LeakyReLU(negative_slope=0.01)), 
    ('Linear2', nn.Linear(256, 256)),
    ('LeakyReLU1', nn.LeakyReLU(negative_slope=0.01)), 
    ('Linear3', nn.Linear(256, 1))
  ])) 
  return model


def generator(noise_dim=NOISE_DIM):
  """
  Build and return a PyTorch nn.Sequential model implementing the architecture in the notebook.
  """
  model = None
  ############################################################################
  # TODO: Implement generator.                                               #
  ############################################################################
  model = nn.Sequential(OrderedDict([
    ('Linear1', nn.Linear(noise_dim, 1024)),
    ('LeakyReLU1', nn.ReLU()), 
    ('Linear2', nn.Linear(1024, 1024)),
    ('LeakyReLU1', nn.ReLU()), 
    ('Linear3', nn.Linear(1024, 784)),
    ('Tanh', nn.Tanh())
  ]))
  return model  

def discriminator_loss(logits_real, logits_fake):
  """
  Computes the discriminator loss described above.
  
  Inputs:
  - logits_real: PyTorch Tensor of shape (N,) giving scores for the real data.
  - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
  
  Returns:
  - loss: PyTorch Tensor containing (scalar) the loss for the discriminator.
  """
  loss = None
  ##############################################################################
  # TODO: Implement discriminator_loss.                                        #
  ##############################################################################
  # note that the logits_real and logits_fake contains raw scores,
  # not squashed by the sigmoid function yet for numerical stability
  # sigmoid is incorporated as part of the cross-entropy loss function
  # binary_cross_entropy_with_logits already incorporated the negated form

  # for discriminator, want to classify all real images as 1, all fake images as 0
  target_real = torch.ones_like(logits_real)
  target_fake_discriminator = torch.zeros_like(logits_fake)

  loss = F.binary_cross_entropy_with_logits(logits_real, target_real) \
         + F.binary_cross_entropy_with_logits(logits_fake, target_fake_discriminator)
  return loss

def generator_loss(logits_fake):
  """
  Computes the generator loss described above.

  Inputs:
  - logits_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
  
  Returns:
  - loss: PyTorch Tensor containing the (scalar) loss for the generator.
  """
  loss = None
  ##############################################################################
  # TODO: Implement generator_loss.                                            #
  ##############################################################################
  # for generator, want discriminator to classify all fake images as real
  target = torch.ones_like(logits_fake)
  loss = torch.nn.functional.binary_cross_entropy_with_logits(logits_fake, target)
  return loss

def get_optimizer(model):
  """
  Construct and return an Adam optimizer for the model with learning rate 1e-3,
  beta1=0.5, and beta2=0.999.
  
  Input:
  - model: A PyTorch model that we want to optimize.
  
  Returns:
  - An Adam optimizer for the model with the desired hyperparameters.
  """
  optimizer = None
  optimizer = optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))
  return optimizer


def ls_discriminator_loss(scores_real, scores_fake):
  """
  Compute the Least-Squares GAN loss for the discriminator.
  
  Inputs:
  - scores_real: PyTorch Tensor of shape (N,) giving scores for the real data.
  - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
  
  Outputs:
  - loss: A PyTorch Tensor containing the loss.
  """
  loss = None
  ##############################################################################
  # TODO: Implement ls_discriminator_loss.                                     #
  ##############################################################################
  N = scores_fake.shape[0]
  a = (scores_real - 1).square() + scores_fake.square()
  b = 1/2 * 1/N
  loss = 1/2 * 1/N * ((scores_real-1).square() + scores_fake.square()).sum()

  return loss

def ls_generator_loss(scores_fake):
  """
  Computes the Least-Squares GAN loss for the generator.
  
  Inputs:
  - scores_fake: PyTorch Tensor of shape (N,) giving scores for the fake data.
  
  Outputs:
  - loss: A PyTorch Tensor containing the loss.
  """
  loss = None
  ##############################################################################
  # TODO: Implement ls_generator_loss.                                         #
  ##############################################################################
  N = scores_fake.shape[0]
  loss = 1/2 * 1/N * (scores_fake-1).square().sum()

  return loss


def build_dc_classifier():
  """
  Build and return a PyTorch nn.Sequential model for the DCGAN discriminator implementing
  the architecture in the notebook.
  """
  model = None
  ############################################################################
  # TODO: Implement build_dc_classifier.                                     #
  ############################################################################
  # spatial dimension: 28x28 -> 24x24 -> 12x12 -> 8x8 -> 4x4
  model = nn.Sequential(OrderedDict([
    ('Unflatten', nn.Unflatten(1, (1, 28,28))),

    ('Conv2d1', nn.Conv2d(1, 32, kernel_size=5, stride=1)),
    ('LeakyReLU1', nn.LeakyReLU(negative_slope=0.01)),
    ('MaxPool1', nn.MaxPool2d(2, stride=2)), 

    ('Conv2d2', nn.Conv2d(32, 64, kernel_size=5, stride=1)),
    ('LeakyReLU2', nn.LeakyReLU(negative_slope=0.01)),
    ('MaxPool2', nn.MaxPool2d(2, stride=2)), 

    ('Flatten', nn.Flatten(start_dim=1)), 
    ('Linear', nn.Linear(4*4*64, 4*4*64)),
    ('LeakyReLU2', nn.LeakyReLU(negative_slope=0.01)),
    ('Linear2', nn.Linear(4*4*64, 1))
  ]))
  return model

def build_dc_generator(noise_dim=NOISE_DIM):
  """
  Build and return a PyTorch nn.Sequential model implementing the DCGAN generator using
  the architecture described in the notebook.
  """
  model = None
  ############################################################################
  # TODO: Implement build_dc_generator.                                      #
  ############################################################################
  model = nn.Sequential(OrderedDict([
    ('Linear1', nn.Linear(noise_dim, 1024)),
    ('ReLU1', nn.ReLU()),
    ('BatchNorm1', nn.BatchNorm1d(num_features=1024)),

    ('Linear2', nn.Linear(1024, 7*7*128)),
    ('ReLU2', nn.ReLU()),
    ('BatchNorm2', nn.BatchNorm1d(num_features=7*7*128)),
    ('Unflatten', nn.Unflatten(1, (128, 7, 7))),

    ('Conv2DT1', nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)), 
    ('ReLU3', nn.ReLU()),
    ('BatchNorm3', nn.BatchNorm2d(num_features=64)),
    ('Conv2DT2', nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1)), 

    ('Tanh', nn.Tanh()),
    ('Flatten', nn.Flatten(start_dim=1))
  ]))

  return model
