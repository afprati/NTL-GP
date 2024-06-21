import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import json
import gpytorch
import pyro
print(gpytorch.__version__) # needs to be 1.8.1
from model.multitaskmodel import MultitaskGPModel
from utilities.savejson import savejson
from utilities.visualize_ntl import visualize_ntl
from utilities.data_prep import data_prep
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
training_iterations = 2 if smoke_test else 0 #70
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
            #with gpytorch.settings.cholesky_jitter(1e-2):
            output = model(x_batch.to(device))
            with gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
                loss = -mll(output, y_batch.to(device))
                loss.backward()
                optimizer.step()
                log_lik += -loss.item()*y_batch.shape[0]
            if j % 50:
                print('Epoch %d Iter %d - Loss: %.3f' % (i + 1, j+1, loss.item()))
        print('Epoch %d - log lik: %.3f' % (i + 1, log_lik))

    return model, likelihood


def ntl(INFERENCE):

    train_x, train_y, test_x, test_y, X_max_v, T0, likelihood, data = data_prep(INFERENCE, 1)
    
    model = MultitaskGPModel(test_x, test_y, X_max_v, likelihood, MAP="MAP" in INFERENCE)
    model.drift_t_module.T0 = T0

    # group effects
    # model.x_covar_module[0].c2 = torch.var(train_y)
    # model.x_covar_module[0].raw_c2.requires_grad = False

    # covariate effects initialize to 0.05**2
    for i in range(len(X_max_v)):
        model.x_covar_module[i].c2 = torch.tensor(0.05**2)


    model.group_mean_module.constantvector.data[0].fill_(0.8) # mean_ntl of control
    model.group_mean_module.constantvector.data[1].fill_(0.6) # mean_ntl of treated

    # set precision to double tensors
    torch.set_default_tensor_type(torch.DoubleTensor)
    train_x, train_y = train_x.to(device), train_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)
    model.to(device)
    likelihood.to(device)

    # define Loss for GPs - the marginal log likelihood
    mll = VariationalELBO(likelihood, model, num_data=train_y.size(0))


    if not os.path.isdir("results"):
        os.mkdir("results")


    transforms = {   
            'group_index_module.raw_rho': model.group_index_module.raw_rho_constraint.transform,
            'group_t_covar_module.base_kernel.raw_lengthscale': model.group_t_covar_module.base_kernel.raw_lengthscale_constraint.transform,
            'group_t_covar_module.raw_outputscale': model.group_t_covar_module.raw_outputscale_constraint.transform,
            'unit_t_covar_module.base_kernel.raw_lengthscale': model.unit_t_covar_module.base_kernel.raw_lengthscale_constraint.transform,
            'unit_t_covar_module.raw_outputscale': model.unit_t_covar_module.raw_outputscale_constraint.transform,
            'likelihood.noise_covar.raw_noise': likelihood.noise_covar.raw_noise_constraint.transform
        }

    priors= {
            'group_index_module.raw_rho': pyro.distributions.Normal(0, 1.5),
            'group_t_covar_module.base_kernel.raw_lengthscale': pyro.distributions.Normal(30, 10).expand([1, 1]),
            'group_t_covar_module.raw_outputscale': pyro.distributions.Normal(-7, 1),
            'unit_t_covar_module.base_kernel.raw_lengthscale': pyro.distributions.Normal(30, 10).expand([1, 1]),
            'unit_t_covar_module.raw_outputscale': pyro.distributions.Normal(-7, 1),
            'likelihood.noise_covar.raw_noise': pyro.distributions.Normal(-7, 1).expand([1]),
            #'x_covar_module.0.raw_c2': pyro.distributions.Normal(-7, 1).expand([1]),
            #'x_covar_module.1.raw_c2': pyro.distributions.Normal(-7, 1).expand([1])
            #'model.x_covar_module.2.raw_c2': pyro.distributions.Normal(-6, 1).expand([1])
        }
    
        
    if INFERENCE=='MAP':
        model.group_index_module._set_rho(0.0)
        model.group_t_covar_module.outputscale = 0.05**2  
        model.group_t_covar_module.base_kernel.lengthscale = 15
        likelihood.noise_covar.noise = 0.05**2
        model.unit_t_covar_module.outputscale = 0.05**2 
        model.unit_t_covar_module.base_kernel.lengthscale = 30

        # covariate effects initialize to 0.01**2
        for i in range(len(X_max_v)):
            model.x_covar_module[i].c2 = torch.tensor(0.05**2)

        for name, param in model.drift_t_module.named_parameters():
            param.requires_grad = True
        
        #model.drift_t_module._set_T1(2.0)  # when treatment starts taking effect
        #model.drift_t_module._set_T2(5.0) # when treatment stabilizes
        model.drift_t_module.base_kernel.lengthscale = 30.0
        model.drift_t_module.outputscale = 0.05**2

        # model.drift_t_module.raw_T1.requires_grad = False
        # model.drift_t_module.raw_T2.requires_grad = False
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model, likelihood = train(test_x, test_y, model, likelihood, mll, optimizer, training_iterations)

        torch.save(model.state_dict(), 'results/ntl_' +  INFERENCE + '_model_state.pth')
        #torch.save(model, 'results/ntl_' +  INFERENCE + '_model_state.pb') # can't pickle lambda on windows
        
        return
        
        model.group_index_module._set_rho(0.9)
        model.group_t_covar_module.outputscale = 0.02**2 
        model.group_t_covar_module.base_kernel._set_lengthscale(3)
        likelihood.noise_covar.noise = 0.03**2
        model.unit_t_covar_module.outputscale = 0.02**2 
        model.unit_t_covar_module.base_kernel._set_lengthscale(30)
       
        # covariate effects initialize to 0.0**2

        for i in range(len(X_max_v)-1):
            model.x_covar_module[i].c2 = torch.tensor(0.01**2)
        #     model.x_covar_module[i].raw_c2.requires_grad = False

        initial_params =  {'group_index_module.rho_prior': model.group_index_module.raw_rho.detach(),\
            'group_t_covar_module.base_kernel.lengthscale_prior':  model.group_t_covar_module.base_kernel.raw_lengthscale.detach(),\
            'group_t_covar_module.outputscale_prior': model.group_t_covar_module.raw_outputscale.detach(),\
            'unit_t_covar_module.base_kernel.lengthscale_prior':  model.unit_t_covar_module.base_kernel.raw_lengthscale.detach(),\
            'unit_t_covar_module.outputscale_prior': model.unit_t_covar_module.raw_outputscale.detach(),\
            'likelihood.noise_covar.noise_prior': likelihood.raw_noise.detach(),
            #'x_covar_module.0.c2_prior': model.x_covar_module[0].raw_c2.detach(),
            #'x_covar_module.1.c2_prior': model.x_covar_module[1].raw_c2.detach()
            }
        return
        
    else:
        model.load_strict_shapes(False)
        state_dict = torch.load('results/ntl_MAP_model_state.pth')
        model.load_state_dict(state_dict)

        print(f'Parameter name: rho value = {model.group_index_module.rho.detach().numpy()}')
        print(f'Parameter name: group ls value = {model.group_t_covar_module.base_kernel.lengthscale.detach().numpy()}')
        print(f'Parameter name: group os value = {np.sqrt(model.group_t_covar_module.outputscale.detach().numpy())}')
        print(f'Parameter name: unit ls value = {model.unit_t_covar_module.base_kernel.lengthscale.detach().numpy()}')
        print(f'Parameter name: unit os value = {np.sqrt(model.unit_t_covar_module.outputscale.detach().numpy())}')
        print(f'Parameter name: noise value = {np.sqrt(likelihood.noise.detach().numpy())}')
        print(f'Parameter name: weekday std value = {np.sqrt(model.x_covar_module[0].c2.detach().numpy())}')
        print(f'Parameter name: day std value = {np.sqrt(model.x_covar_module[1].c2.detach().numpy())}')
        print(f'Parameter name: unit std value = {np.sqrt(model.x_covar_module[2].c2.detach().numpy())}')
        
        print(f'Parameter name: drift ls value = {model.drift_t_module.base_kernel.lengthscale.detach().numpy()}')
        print(f'Parameter name: drift cov os value = {np.sqrt(model.drift_t_module.outputscale.detach().numpy())}')


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