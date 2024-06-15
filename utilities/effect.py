import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import gpytorch
#print(gpytorch.__version__) # needs to be 1.8.1
from model.multitaskmodel import MultitaskGPModel
#from utilities.visualize import plot_posterior, plot_pyro_posterior,plot_pyro_prior
#from utilities.visualize import visualize, plot_prior
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
ohe = OneHotEncoder()
ohe = LabelEncoder()
X = data.drop(columns=["obs_id", "date", "mean_ntl", "Treated", "post","period"]).to_numpy().reshape(-1,) # , "weekday","affiliation","callsign"
Group = data.Treated.to_numpy().reshape(-1,1)
ohe.fit(X)
X = ohe.transform(X)
station_le = LabelEncoder()
ids = data.obs_id.to_numpy().reshape(-1,)
station_le.fit(ids)
ids = station_le.transform(ids)
# weekday/day/unit effects and time trend
X = np.concatenate((X.reshape(ds.shape[0],-1),ds,ids.reshape(-1,1),Group,ds), axis=1)
# numbers of dummies for each effect
X_max_v = [np.max(X[:,i]).astype(int) for i in range(X.shape[1]-2)]

Y = data.mean_ntl.to_numpy()
T0 = data[data.date==datetime.date(1999, 1, 1)].period.to_numpy()[0]
train_condition = (data.post!=1) | (data.Treated!=1)
device = torch.device('cpu')
train_x = torch.Tensor(X[train_condition], device=device).double()
train_y = torch.Tensor(Y[train_condition], device=device).double()
train_data = data[train_condition]

idx = data.Treated.to_numpy()
train_g = torch.from_numpy(idx[train_condition]).to(device)

test_x = torch.Tensor(X).double()
test_y = torch.Tensor(Y).double()
test_g = torch.from_numpy(idx)

model = MultitaskGPModel(test_x, test_y, X_max_v, likelihood, MAP="MAP")
model.drift_t_module.T0 = T0
model2 = MultitaskGPModel(train_x, train_y, X_max_v, likelihood, MAP="MAP")

model.load_strict_shapes(False)
model2.load_strict_shapes(False)
state_dict = torch.load('C:/Users/miame/OneDrive/Backups/Documents/GitHub/NTL-GP/results/ntl_MAP_model_state.pth')
model.load_state_dict(state_dict)
model2.load_state_dict(state_dict)

with torch.no_grad(), gpytorch.settings.fast_pred_var():
    out = likelihood(model(train_x))
    mu_f = out.mean
    V = out.covariance_matrix
    L = torch.linalg.cholesky(V, upper=False)
    lower, upper = out.confidence_region()


results = pd.DataFrame({"gpr_mean":mu_f})
results['true_y'] = train_y
results['gpr_lwr'] = lower
results['gpr_upr'] = upper
results.to_csv("./results/ntl_fitted_gpr.csv",index=False)
train_data.to_csv("./results/ntl_train_data.csv",index=False)