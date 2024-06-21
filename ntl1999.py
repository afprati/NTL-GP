import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import json
import gpytorch
print(gpytorch.__version__) # needs to be 1.8.1
import pyro
from pyro.infer.mcmc import NUTS, MCMC, HMC
from model.multitaskmodel import MultitaskGPModel
from utilities.savejson import savejson
from utilities.visualize import plot_posterior, plot_pyro_posterior,plot_pyro_prior
from utilities.visualize_ntl import visualize_ntl
from utilities.data_prep import data_prep
from utilities.synthetic import generate_synthetic_data
from model.fixedeffect import TwoWayFixedEffectModel
import pandas as pd
import numpy as np
import argparse
import datetime
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import dill as pickle

from gpytorch.mlls import VariationalELBO
from torch.utils.data import TensorDataset, DataLoader

torch.manual_seed(123)

smoke_test = ('CI' in os.environ)
training_iterations = 2 if smoke_test else 20 #70
num_samples = 2 if smoke_test else 10 #500
warmup_steps = 2 if smoke_test else 10 #500
load_batch_size = 512 # can also be 256

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

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
            output = model(x_batch)
            with gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
                loss = -mll(output, y_batch)
                loss.backward()
                optimizer.step()
                log_lik += -loss.item()*y_batch.shape[0]
            if j % 20 == 0:
                print('Epoch %d Iter %d - Loss: %.3f' % (i + 1, j+1, loss.item()))
        print('Epoch %d - log lik: %.3f' % (i + 1, log_lik))

    return model, likelihood


def ntl(INFERENCE):
    train_x, train_y, test_x, test_y, X_max_v, T0, likelihood = data_prep(INFERENCE)
    
    model = MultitaskGPModel(test_x, test_y, X_max_v, likelihood, T0, MAP="MAP" in INFERENCE)

    # group effects
    # model.x_covar_module[0].c2 = torch.var(train_y)
    # model.x_covar_module[0].raw_c2.requires_grad = False

    # covariate effects initialize to 0.05**2
    for i in range(len(X_max_v)):
        model.x_covar_module[i].c2 = torch.tensor(0.05**2)

    # fix unit mean/variance by not requiring grad
    #model.x_covar_module[-1].raw_c2.requires_grad = False

    # model.unit_mean_module.constant.data.fill_(0.12)
    # model.unit_mean_module.constant.requires_grad = False
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
            'likelihood.noise_covar.raw_noise': likelihood.noise_covar.raw_noise_constraint.transform,
            #'x_covar_module.0.raw_c2': model.x_covar_module[0].raw_c2_constraint.transform,
            #'x_covar_module.1.raw_c2': model.x_covar_module[1].raw_c2_constraint.transform
            #'x_covar_module.2.raw_c2': model.x_covar_module[2].raw_c2_constraint.transform
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
    

    def pyro_model(x, y):
        
        fn = pyro.random_module("model", model, prior=priors)
        sampled_model = fn()
        
        output = sampled_model.likelihood(sampled_model(x))
        pyro.sample("obs", output, obs=y)
    
        with open('results/ntl_MCMC.pkl', 'rb') as f:
            mcmc_run = pickle.load(f)
        mcmc_samples = mcmc_run.get_samples()
        print(mcmc_run.summary())
        plot_pyro_posterior(mcmc_samples, transforms)
        # plot_posterior(mcmc_samples)
        return
        for k, d in mcmc_samples.items():
            mcmc_samples[k] = d[idx]
        model.pyro_load_from_samples(mcmc_samples)
    
        return
        
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

        with gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
            nuts_kernel = NUTS(pyro_model, adapt_step_size=True, adapt_mass_matrix=True, jit_compile=False,\
                init_strategy=pyro.infer.autoguide.initialization.init_to_value(values=initial_params))
            hmc_kernel = HMC(pyro_model, step_size=0.1, num_steps=10, adapt_step_size=True,\
                init_strategy=pyro.infer.autoguide.initialization.init_to_mean())
            mcmc_run = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps)
            mcmc_run.run(train_x, train_y)
            pickle.dump(mcmc_run, open("results/ntl_MCMC.pkl", "wb"))
            # plot_pyro_posterior(mcmc_run.get_samples(), transforms)

        return

        
    else:
        model.load_strict_shapes(False)
        state_dict = torch.load('results/ntl_MAP_model_state.pth')
        model.load_state_dict(state_dict)

        print(f'Parameter name: rho value = {model.group_index_module.rho.detach().numpy()}')
        # print(f'Parameter name: unit mean value = {model.unit_mean_module.constant.detach().numpy()}')
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

        #visualize_ntl(data, test_x, test_y, test_g, model, model2, likelihood, T0, obs_le, train_condition)

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