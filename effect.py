import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import gpytorch
#print(gpytorch.__version__) # needs to be 1.8.1
from model.multitaskmodel import MultitaskGPModel
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import TensorDataset, DataLoader
from utilities.data_prep import data_prep
load_batch_size = 512 # can also be 256


torch.manual_seed(123)

# define likelihood
noise_prior = gpytorch.priors.GammaPrior(concentration=1,rate=10)
likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=noise_prior, noise_constraint=gpytorch.constraints.Positive())

train_x, train_y, T0, likelihood = data_prep("MAP")

model = MultitaskGPModel(train_x, train_y, likelihood, T0, MAP="MAP")

model.load_strict_shapes(False)
state_dict = torch.load('./results/ntl_MAP_model_state.pth')
model.load_state_dict(state_dict)


# finding the posterior, using the train data
# setting to eval mode
model.eval()
likelihood.eval()

train_dataset = TensorDataset(train_x, train_y)
train_loader = DataLoader(train_dataset, batch_size=load_batch_size, shuffle=False)
mu_f_full = []
lower_full = []
upper_full = []


with torch.no_grad(), gpytorch.settings.fast_pred_var():
    for j, (x_batch, y_batch) in enumerate(train_loader):
        print(j)
        out = model(x_batch)
        mu_f = out.mean
        lower, upper = out.confidence_region()
        mu_f_full.append(mu_f)
        lower_full.append(lower)
        upper_full.append(upper)

mu_f_np = torch.concat(mu_f_full, dim=0).numpy()
lower_np = torch.concat(lower_full, dim=0).numpy()
upper_np = torch.concat(upper_full, dim=0).numpy()

RMSE = np.square(mu_f_np - train_y.numpy()).mean()**0.5
print(RMSE)

# finding effect/counterfactuals
test_x0 = train_x.clone().detach().requires_grad_(False)
test_x0[:,-1] = 0

train_dataset = TensorDataset(test_x0, train_y)
train_loader = DataLoader(train_dataset, batch_size=load_batch_size, shuffle=False)
out0_full = []
lower0_full = []
upper0_full = []


with torch.no_grad(), gpytorch.settings.fast_pred_var():
    for j, (x_batch, y_batch) in enumerate(train_loader):
        print(j)
        out0 = model(x_batch)
        lower0, upper0 = out0.confidence_region()
        out0_full.append(out0.mean)
        lower0_full.append(lower0)
        upper0_full.append(upper0)


out0_np = torch.concat(out0_full, dim=0).numpy()
lower0_np = torch.concat(lower0_full, dim=0).numpy()
upper0_np = torch.concat(upper0_full, dim=0).numpy()

mask = (train_x[:,-1]==1)
effect = mu_f_np[mask].mean() - out0_np[mask].mean()
effect_std = np.sqrt((mu_f_np[mask].var() + out0_np[mask].var())) / np.sqrt(train_x.numpy()[mask].shape[0])
#BIC = (2+4+6+1)*torch.log(torch.tensor(train_x.size()[0])) + 2*loss*train_x.size()[0]
print("ATT: {:0.3f} +- {:0.3f}\n".format(effect, effect_std))
#print("model evidence: {:0.3f} \n".format(-loss*train_x.size()[0]))
#print("BIC: {:0.3f} \n".format(BIC))


results = pd.DataFrame({"gpr_mean":mu_f_np})
results['true_y'] = train_y
results['gpr_lwr'] = lower_np
results['gpr_upr'] = upper_np
results['t0_mean'] = out0_np
results['t0_lwr'] = lower0_np
results['t0_upr'] = upper0_np

data = pd.read_csv("data/data1999.csv",index_col=[0])
results['Treated'] = data.set_index(results.index)['Treated']
results['obs_id'] = data.set_index(results.index)['obs_id']
results['period'] = data.set_index(results.index)['period']

results.to_csv("./results/ntl_fitted_gpr.csv",index=False)


print('done')
