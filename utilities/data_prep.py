import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import gpytorch
#print(gpytorch.__version__) # needs to be 1.8.1
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder


device = torch.device('cpu')

def data_prep(INFERENCE):
    data = pd.read_csv("data/data1999.csv",index_col=[0])
    
    device = torch.device('cpu')
    torch.set_default_tensor_type(torch.DoubleTensor)

    # preprocess data
    data = data[~data.obs_id.isin([867, 1690])]
    data.date = data.date.apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y').date())

    ds = data['period'].to_numpy().reshape((-1,1))
    ohe = LabelEncoder()
    X = data.drop(columns=["obs_id", "date", "mean_ntl", "Treated",
    "post","period"]).to_numpy().reshape(-1,) # econ_active_rate, literacy_rate, pop_hh_ratio, hindu_rate, polygamy_rate
    Group = data.Treated.to_numpy().reshape(-1,1)
    ohe.fit(X)
    X = ohe.transform(X)
    obs_le = LabelEncoder()
    ids = data.obs_id.to_numpy().reshape(-1,)
    obs_le.fit(ids)
    ids = obs_le.transform(ids)
    print(ids)
    # covariates and time trend
    X = np.concatenate((X.reshape(ds.shape[0],-1),ids.reshape(-1,1),Group,ds), axis=1)
    # numbers of dummies for each effect
    X_max_v = [np.max(X[:,i]).astype(int) for i in range(X.shape[1]-2)]

    Y = data.mean_ntl.to_numpy()
    T0 = data[data.date==datetime.date(1999, 1, 1)].period.to_numpy()[0]
    train_condition = (data.post!=1) | (data.Treated!=1)
    train_x = torch.Tensor(X[train_condition], device=device).double()
    train_y = torch.Tensor(Y[train_condition], device=device).double()

    idx = data.Treated.to_numpy()
    #train_g = torch.from_numpy(idx).to(device)

    test_x = torch.Tensor(X).double()
    test_y = torch.Tensor(Y).double()
    #test_g = torch.from_numpy(idx)
    
    # define likelihood
    noise_prior = gpytorch.priors.GammaPrior(concentration=1,rate=10)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=noise_prior if "MAP" in INFERENCE else None,\
            noise_constraint=gpytorch.constraints.Positive())
    print("data loaded")
    
    return train_x, train_y, test_x, test_y, X_max_v, T0, likelihood