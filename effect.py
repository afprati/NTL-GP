import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import gpytorch
#print(gpytorch.__version__) # needs to be 1.8.1
from model.multitaskmodel import MultitaskGPModel
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import OneHotEncoder, LabelEncoder


torch.manual_seed(123)

# define likelihood
noise_prior = gpytorch.priors.GammaPrior(concentration=1,rate=10)
likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=noise_prior, noise_constraint=gpytorch.constraints.Positive())

# preprocess data
data = pd.read_csv("C:/Users/miame/OneDrive/Backups/Documents/GitHub/NTL-GP/data/data1999test.csv",index_col=[0])
data = data[~data.obs_id.isin([867, 1690])]
print(data.shape)
N = data.obs_id.unique().shape[0]
data.date = data.date.apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y').date())

ds = data['period'].to_numpy().reshape((-1,1))
ohe = LabelEncoder()
X = data.drop(columns=["obs_id", "date", "mean_ntl", "Treated",
"post","period"]).to_numpy().reshape(-1,) 
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
train_data = data

idx = data.Treated.to_numpy()
train_g = torch.from_numpy(idx).to(device)

test_x = torch.Tensor(X).double()
test_y = torch.Tensor(Y).double()
test_g = torch.from_numpy(idx)


model = MultitaskGPModel(test_x, test_y, X_max_v, likelihood, MAP="MAP")
model.drift_t_module.T0 = T0

model.load_strict_shapes(False)
state_dict = torch.load('C:/Users/miame/OneDrive/Backups/Documents/GitHub/NTL-GP/results/ntl_MAP_model_state.pth')
model.load_state_dict(state_dict)


# finding the posterior, using the train data
# setting to eval mode
model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    model(train_x)
    out = likelihood(model(train_x))
    mu_f = out.mean
    V = out.covariance_matrix
    L = torch.linalg.cholesky(V, upper=False)
    lower, upper = out.confidence_region()

RMSE = np.square((out.mean - train_y).detach().numpy()).mean()**0.5
print(RMSE)

# finding effect

test_x1 = train_x.clone().detach().requires_grad_(False)
test_x1[:,-2] = 1
test_x0 = train_x.clone().detach().requires_grad_(False)
test_x0[:,-2] = 0

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    out1 = likelihood(model(test_x1))
    out0 = likelihood(model(test_x0))
    V1 = out1.covariance_matrix
    V0 = out0.covariance_matrix
    L0 = torch.linalg.cholesky(V0, upper=False)
    lower1, upper1 = out1.confidence_region()
    lower0, upper0 = out0.confidence_region()
    
effect = out1.mean.numpy().mean() - out0.mean.numpy().mean()
effect_std = np.sqrt((out1.mean.numpy().mean()+out0.mean.numpy().mean())) / np.sqrt(train_x.size()[0])
#BIC = (2+4+6+1)*torch.log(torch.tensor(train_x.size()[0])) + 2*loss*train_x.size()[0]
print("ATE: {:0.3f} +- {:0.3f}\n".format(effect, effect_std))
#print("model evidence: {:0.3f} \n".format(-loss*train_x.size()[0]))
#print("BIC: {:0.3f} \n".format(BIC))

results = pd.DataFrame({"gpr_mean":mu_f})
results['true_y'] = train_y
results['gpr_lwr'] = lower
results['gpr_upr'] = upper
results['t1_mean'] = out1.mean
results['t1_lwr'] = lower1
results['t1_upr'] = upper1
results['t0_mean'] = out0.mean
results['t0_lwr'] = lower0
results['t0_upr'] = upper0
results['period'] = train_data['period']
results.to_csv("C:/Users/miame/OneDrive/Backups/Documents/GitHub/NTL-GP/results/ntl_fitted_gpr.csv",index=False)
train_data.to_csv("C:/Users/miame/OneDrive/Backups/Documents/GitHub/NTL-GP/results/ntl_train_data.csv",index=False)


print('done')
