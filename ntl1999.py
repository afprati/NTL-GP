import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import json
import gpytorch
import pyro
print(gpytorch.__version__) # needs to be 1.8.1
from utilities.savejson import savejson
from utilities.visualize_ntl import visualize_ntl
from utilities.data_prep import data_prep, model_prep
import pandas as pd
import numpy as np
import argparse
import datetime
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from gpytorch.mlls import VariationalELBO
from torch.utils.data import TensorDataset, DataLoader
import dill as pickle

torch.manual_seed(123)

smoke_test = ('CI' in os.environ)
training_iterations = 2 if smoke_test else 100 #70
num_samples = 2 if smoke_test else 10 #500
warmup_steps = 2 if smoke_test else 10 #500
load_batch_size = 512 # can also be 256

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#device = torch.device('cpu')

def train(train_x, train_y, model, likelihood, mll, optimizer, training_iterations):
    # Find optimal model hyperparameters
    
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=load_batch_size, shuffle=True)

    model.train()
    likelihood.train()
    print("start training...")
    for i in range(training_iterations):
        log_lik = 0
        for j, (x_batch, y_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            output = model(x_batch.to(device))
            with gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
                loss = -mll(output, y_batch.to(device))
                loss.backward()
                optimizer.step()
                log_lik += -loss.item()*y_batch.shape[0]
            if j % 20 == 0:
                print('Epoch %d Iter %d - Loss: %.3f' % (i + 1, j+1, loss.item()))
        print('Epoch %d - log lik: %.3f' % (i + 1, log_lik))

    return model, likelihood


def ntl(INFERENCE):
    train_x, train_y, T0, likelihood = data_prep(INFERENCE)
    
    model = model_prep(train_x, train_y, likelihood, T0, MAP="MAP"==INFERENCE)

    model.group_mean_module.constantvector.data[0].fill_(train_y[train_x[:,-3]==0].mean()) # mean_ntl of control
    model.group_mean_module.constantvector.data[1].fill_(train_y[train_x[:,-3]==1].mean()) # mean_ntl of treated

    # set precision to double tensors
    torch.set_default_tensor_type(torch.DoubleTensor)


    # define Loss for GPs - the marginal log likelihood
    mll = VariationalELBO(likelihood, model, num_data=train_y.size(0))

    if not os.path.isdir("results"):
        os.mkdir("results")
    
        
    if INFERENCE=='MAP':
        model.group_index_module._set_rho(0.5)
        model.group_t_covar_module.outputscale = 1**2  
        model.group_t_covar_module.base_kernel.lengthscale = 3
        likelihood.noise_covar.noise = 0.5**2
        model.unit_t_covar_module.outputscale = 1**2  
        model.unit_t_covar_module.base_kernel.lengthscale = 3

        for name, param in model.drift_t_module.named_parameters():
            param.requires_grad = True
        
        model.drift_t_module.base_kernel.lengthscale = 5.0
        model.drift_t_module.outputscale = 0.25**2  
 
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        model, likelihood = train(train_x, train_y, model, likelihood, mll, optimizer, training_iterations)

        torch.save(model.state_dict(), 'results/ntl_' +  INFERENCE + '_model_state.pth')
     

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='python ntl1999.py --type lights --inference MAP')
    parser.add_argument('-t','--type', help='ntl/synthetic', required=True)
    parser.add_argument('-i','--inference', help='MCMC/MAP/MAPLOAD/MCMCLOAD', required=True)
    args = vars(parser.parse_args())
    if args['type'] == 'lights':
        print("starting NTL")
        ntl(INFERENCE=args['inference'])
    else:
        exit()