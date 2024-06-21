import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
import gpytorch
#print(gpytorch.__version__) # needs to be 1.8.1
import pandas as pd
import numpy as np
import datetime
from sklearn.preprocessing import LabelEncoder



def data_prep(INFERENCE, training):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    #device = torch.device('cpu')

    data = pd.read_csv("data/data1999test.csv",index_col=[0])

    torch.set_default_tensor_type(torch.DoubleTensor)

    # preprocess data
    data = data[~data.obs_id.isin([867, 1690])]
    data.date = data.date.apply(lambda x: datetime.datetime.strptime(x, '%m/%d/%Y').date())

    ds = data['period'].to_numpy().reshape((-1,1))
    ohe = LabelEncoder()
    X = data.drop(columns=["obs_id", "date", "mean_ntl", "Treated",\
    "post","period"]).to_numpy().reshape(-1,) # econ_active_rate, literacy_rate, pop_hh_ratio, hindu_rate, polygamy_rate
    Group = data.Treated.to_numpy().reshape(-1,1)
    ohe.fit(X)
    X = ohe.transform(X)
    obs_le = LabelEncoder()
    ids = data.obs_id.to_numpy().reshape(-1,)
    obs_le.fit(ids)
    ids = obs_le.transform(ids)

    # covariates and time trend
    T0 = data[data.date==datetime.date(1999, 1, 1)].period.to_numpy()[0]
    if training==1:
        train_condition = (data.post!=1) | (data.Treated!=1) # mask for training
        train_x = torch.tensor(X[train_condition], device=device).double()
        train_y = torch.tensor(Y[train_condition], device=device).double()

    else:
        train_x = torch.tensor(X, device=device).double()
        train_y = torch.tensor(Y, device=device).double()


    idx = data.Treated.to_numpy()
    #train_g = torch.from_numpy(idx).to(device)

    test_x = torch.tensor(X).double()
    test_y = torch.tensor(Y).double()
    #test_g = torch.from_numpy(idx)

    # define likelihood
    noise_prior = gpytorch.priors.GammaPrior(concentration=1,rate=10)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_prior=noise_prior if "MAP" in INFERENCE else None,\
            noise_constraint=gpytorch.constraints.Positive())
    print("data loaded")

    return train_x, train_y, test_x, test_y, X_max_v, T0, likelihood, data
