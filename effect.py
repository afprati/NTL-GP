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

data = pd.read_csv("data/data1999.csv",index_col=[0])

#data_prep(data)

# define likelihood
noise_prior = gpytorch.priors.GammaPrior(concentration=1,rate=10)
likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=noise_prior, noise_constraint=gpytorch.constraints.Positive())

# preprocess data
data = pd.read_csv("C:/Users/miame/OneDrive/Backups/Documents/GitHub/NTL-GP/data/data1999.csv",index_col=[0])
data = data[~data.obs_id.isin([867, 1690])]
mask = (data.post==1) & (data.Treated==1)
print(data.shape)
N = data.obs_id.unique().shape[0]
data.date = data.date.apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y').date())

ds = data['period'].to_numpy().reshape((-1,1))
ohe = LabelEncoder()
X = data.drop(columns=["obs_id", "date", "mean_ntl", "Treated", "post","period"]).to_numpy().reshape(-1,) 
Group = data.Treated.to_numpy().reshape(-1,1)
ohe.fit(X)
X = ohe.transform(X)
obs_le = LabelEncoder()
ids = data.obs_id.to_numpy().reshape(-1,)
obs_le.fit(ids)
ids = obs_le.transform(ids)
print(ids)
X = np.concatenate((X.reshape(ds.shape[0],-1),ids.reshape(-1,1),Group,ds), axis=1)
X_max_v = [np.max(X[:,i]).astype(int) for i in range(X.shape[1]-2)]

Y = data.mean_ntl.to_numpy()
T0 = data[data.date==datetime.date(1999, 1, 1)].period.to_numpy()[0]
device = torch.device('cpu')
train_x = torch.Tensor(X, device=device).double()
train_y = torch.Tensor(Y, device=device).double()

#idx = data.Treated.to_numpy()
#train_g = torch.from_numpy(idx).to(device)

#test_x = torch.Tensor(X).double()
test_y = torch.Tensor(Y).double()
#test_g = torch.from_numpy(idx)


model = MultitaskGPModel(train_x, train_y, X_max_v, likelihood, MAP="MAP")
model.drift_t_module.T0 = T0

model.load_strict_shapes(False)
state_dict = torch.load('C:/Users/miame/OneDrive/Backups/Documents/GitHub/NTL-GP/results/ntl_MAP_model_state.pth')
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
        out = likelihood(model(x_batch))
        mu_f = out.mean
        lower, upper = out.confidence_region()
        mu_f_full.append(mu_f)
        lower_full.append(lower)
        upper_full.append(upper)

mu_f_np = torch.concatenate(mu_f_full, dim=0).numpy()
lower_np = torch.concatenate(lower_full, dim=0).numpy()
upper_np = torch.concatenate(upper_full, dim=0).numpy()

RMSE = np.square(mu_f_np - train_y.numpy()).mean()**0.5
print(RMSE)

# finding effect/counterfactuals
test_x0 = train_x.clone().detach().requires_grad_(False)
test_x0[:,-2] = 0

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


out0_np = torch.concatenate(out0_full, dim=0).numpy()
lower0_np = torch.concatenate(lower0_full, dim=0).numpy()
upper0_np = torch.concatenate(upper0_full, dim=0).numpy()


    
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
results['Treated'] = data.set_index(results.index)['Treated']
results['obs_id'] = data.set_index(results.index)['obs_id']
results['period'] = data.set_index(results.index)['period']

results.to_csv("C:/Users/miame/OneDrive/Backups/Documents/GitHub/NTL-GP/results/ntl_fitted_gpr.csv",index=False)




print('done')
