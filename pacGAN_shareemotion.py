# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 10:02:07 2022

@author: shareemotion
"""


import torch
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import TensorDataset, DataLoader
from torch import autograd
import time as t
import matplotlib.pyplot as plt
import os
# from utils.tensorboard_logger import Logger
# from itertools import chain
import numpy as np
import pandas as pd


packing_size = 2


class Generator(torch.nn.Module):
    def __init__(self, channels):
        super().__init__()
        
        self.layer1 = torch.nn.Linear(dim_in, window_size)
        self.relu = torch.nn.ReLU()
        self.layer2 = torch.nn.Linear(window_size, dim_out)
        self.main_module = torch.nn.Sequential(
            # z for latent vector 100
            torch.nn.Linear(dim_z, 1024),
            torch.nn.BatchNorm1d(1024),
            torch.nn.ReLU(True),
            torch.nn.Linear(1024, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(True),
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(True),
            torch.nn.Linear(256, channels)
            )
        
        self.output = torch.nn.Tanh()
        
    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

 
class Discriminator(torch.nn.Module):
    def __init__(self, dim_in, packing_size):
        super().__init__()        
        
        self.dim_in = dim_in * packing_size

        self.main_module = torch.nn.Sequential(            
            torch.nn.Linear(self.dim_in, int(self.dim_in / 2)),
            torch.nn.BatchNorm1d(int(self.dim_in / 2)),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(int(self.dim_in / 2), int(self.dim_in / 4)),
            torch.nn.BatchNorm1d(int(self.dim_in / 4)),
            torch.nn.LeakyReLU(0.2, inplace=True),
            torch.nn.Linear(int(self.dim_in / 4), int(self.dim_in / 8)),
            torch.nn.BatchNorm1d(int(self.dim_in / 8)),
            torch.nn.LeakyReLU(0.2, inplace=True),           
            )
        self.output = torch.nn.Softmax(dim=1)
        
        
    def forward(self, x):
        x = self.main_module(x)
        return self.output(x)

def console(df):
    for i in range(2):
        print(f"df size {i}: {df.size(i)}")


def convert_torch_tensor(df):
    return torch.tensor(np.array(df))

# real_data = real_data
def calculate_gradient_penalty(real_data, fake_data):
    eta = torch.FloatTensor(batch_size, 1).uniform_(0, 1)
    eta_expand = eta.expand(batch_size, real_data.size(1))

    # console(eta_expand)

    interpolate = eta_expand * real_data + ((1 - eta_expand) * fake_data)

    interpolate = Variable(interpolate, requires_grad = True)

    # console(interpolate)

    prob_interpolated = D(interpolate)

    gradients = autograd.grad(outputs=prob_interpolated, 
                                inputs=interpolate, 
                                grad_outputs=torch.ones(
                                    prob_interpolated.size()), 
                                    create_graph=True, 
                                    retain_graph=True
                                )[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * lambda_term
    
    return gradient_penalty


def load_model(D_model_filename, G_model_filename):
    D_model_path = os.path.join(os.getcwd(), D_model_filename)
    G_model_path = os.path.join(os.getcwd(), G_model_filename)
    D.load_state_dict(torch.load(D_model_path))
    G.load_state_dict(torch.load(G_model_path))
    print('Generator model loaded from {}.'.format(G_model_path))
    print('Discriminator model loaded from {}-'.format(D_model_path))



def packing_z(batch_size, z_size):
    return torch.cat(torch.rand((batch_size, z_size)), torch.rand((batch_size, z_size)), axis = 1)






SAVE_PER_TIMES = 100

learning_rate = 1e-4
b1 = 0.5
b2 = 0.999
critic_iter = 10
generator_iters = 10
lambda_term = 10


#  ?????? ??????
batch_size = 3
dim_in = 64
window_size = 2
dim_out = 1
dim_z = 100
nrow = 200
z_size = 100

G = Generator(dim_in)
D = Discriminator(dim_in, packing_size)

# create minority data
rng = np.random.default_rng()
df = pd.DataFrame(rng.random(size=(nrow, dim_in)), columns=list(range(dim_in)))
# df['Y'] = np.random.rand(nrow, 1)
df['Y'] = np.ones((nrow, 1))


# df.columns
data = convert_torch_tensor(df.drop('Y', axis = 1))

dataset = TensorDataset(data.float(), convert_torch_tensor(df['Y']))

dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)



d_opt = optim.Adam(D.parameters(), lr = learning_rate, betas=(b1, b2))
g_opt = optim.Adam(G.parameters(), lr = learning_rate, betas=(b1, b2))






one = torch.tensor(1, dtype=torch.float)
mone = one * -1


st = t.time()


for g_iter in range(generator_iters):
    # Requires grad, Generator requires_grad = False
    for p in D.parameters():
        p.requires_grad = True

    d_loss_real = 0
    d_loss_fake = 0
    Wasserstein_D = 0
    # Train Dicriminator forward-loss-backward-update self.critic_iter times 
    # while 1 Generator forward-loss-backward-update
    for d_iter in range(critic_iter):
        D.zero_grad()

        z1 = torch.rand((batch_size, z_size))
        z2 = torch.rand((batch_size, z_size))

        #packing real_data

        real_data1, train_labels = next(iter(dl))
        real_data2, train_labels = next(iter(dl))

        real_data_packed = torch.cat((real_data1.data, real_data2.data), dim=1)
        print(f"real_data_packed : {real_data_packed.shape}")

        #packing fake_data
        
        fake_data1 = G(z1)
        fake_data2 = G(z2)
      
        fake_data_packed = torch.cat((fake_data1.data, fake_data2.data), dim=1)

        


        # if real_data.size()[0] != batch_size:
        #     continue


        d_loss_real = D(real_data_packed).mean()
        d_loss_real.backward(mone)


        # fake_data = G(z)
        d_loss_fake = D(fake_data_packed).mean()
        d_loss_fake.backward(one)

        gradient_penalty = calculate_gradient_penalty(real_data_packed.data, fake_data_packed.data)
        gradient_penalty.backward()


        d_loss = d_loss_fake - d_loss_real + gradient_penalty
        Wasserstein_D = d_loss_real - d_loss_fake
        d_opt.step()

        print(f'  Discriminator iteration: {d_iter}/{critic_iter}, loss_fake: {d_loss_fake}, loss_real: {d_loss_real}')

     # Generator update
    for p in D.parameters():
        p.requires_grad = False  # to avoid computation


    G.zero_grad()

    z1 = torch.rand((batch_size, z_size))
    z2 = torch.rand((batch_size, z_size))

    fake_data_packed = torch.cat((G(z1), G(z2)), axis = 1)

    g_loss = D(fake_data_packed).mean()

    g_loss.backward(mone)
    g_opt.step()

    print(f'Generator iteration: {g_iter}/{generator_iters}, g_loss: {g_loss}')

    if (g_iter) % SAVE_PER_TIMES == 0:
        # torch.save(G.state_dict(), './generator.pkl')
        # torch.save(D.state_dict(), './discriminator.pkl')
        print('Models save to ./generator.pkl & ./discriminator.pkl ')

end = t.time()
print(f"elapsed time : {end - st} s")
















