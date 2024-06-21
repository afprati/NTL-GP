import torch
import gpytorch
from torch.nn import ModuleList
import json
import numpy as np
from model.customizedkernel import myIndexKernel, constantKernel, myIndicatorKernel
from model.customizedkernel import ConstantVectorMean, DriftScaleKernel, DriftIndicatorKernel
from gpytorch.variational import CholeskyVariationalDistribution, VariationalStrategy
from gpytorch.kernels import ScaleKernel, RBFKernel

class MultitaskGPModel(gpytorch.models.ApproximateGP):

    def __init__(self, train_x, train_y, X_max_v, likelihood, T0=0, MAP=True, seed=123):
        '''
        Inputs:
            - train_x:
            - train_y:
            - likelihood:
        '''
        # setting seed -- should be an argument for the function
        torch.manual_seed(seed)

        # Sparse Variational Formulation
        num_inducing = 500 #can change, higher more accurate but takes longer
        inducing_points = train_x[np.random.choice(train_x.size(0),num_inducing,replace=False),:]
        # trying to make sure inducing points do not have too many zeros; remove duplicates
        inducing_points = torch.unique(inducing_points, dim=0) # make rowwise unique
        #inducing_points += np.random.normal(size=inducing_points.shape)*1e-2
        q_u = CholeskyVariationalDistribution(inducing_points.size(0))
        q_f = VariationalStrategy(self, inducing_points, q_u, \
                                 learn_inducing_locations=False)
        super().__init__(q_f)

        # define priors
        outputscale_prior = gpytorch.priors.GammaPrior(concentration=1,rate=10)
        lengthscale_prior = gpytorch.priors.GammaPrior(concentration=4,rate=1/5)
        rho_prior = gpytorch.priors.UniformPrior(-1, 1)
        unit_outputscale_prior = gpytorch.priors.GammaPrior(concentration=1,rate=10)
        unit_lengthscale_prior = gpytorch.priors.GammaPrior(concentration=4,rate=1/5)
        drift_outputscale_prior = gpytorch.priors.GammaPrior(concentration=1,rate=20)
        drift_lengthscale_prior = gpytorch.priors.GammaPrior(concentration=5,rate=1/5)
        
        # treatment/control groups
        self.num_groups = 2 
        self.num_units = len(train_x[:,-3].unique()) # unit index
        self.T0 = T0

        # covariate effects
        self.X_max_v = X_max_v
        # dim of covariates
        self.d = list(train_x.shape)[1] - 3 # to correspond to unit index

        # same mean of unit bias for all units, could extend this to be unit-dependent
        # self.unit_mean_module = gpytorch.means.ConstantMean()
        self.unit_mean_module =  ConstantVectorMean(d=self.num_units)
        self.group_mean_module = ConstantVectorMean(d=self.num_groups)

        # marginalize weekday/day/unit id effects
        #self.x_covar_module = ModuleList([constantKernel(num_tasks=v+1) for v in self.X_max_v])
        self.x_covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(active_dims=torch.arange(self.d)),
            outputscale_prior=outputscale_prior if MAP else None,
            lengthscale_prior=lengthscale_prior if MAP else None
        )

        # group-level time trend
        self.group_t_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(\
                active_dims=torch.tensor([self.d + 2]),\
                     lengthscale_prior=lengthscale_prior if MAP else None),\
                    outputscale_prior=outputscale_prior if MAP else None) # +2 since last column corresponds to time

        # indicator covariances -- only needed for binary covariates
        #self.x_indicator_module = ModuleList([myIndicatorKernel(num_tasks=v+1) for v in X_max_v])
        self.group_index_module = myIndexKernel(num_tasks=self.num_groups,\
             rho_prior=rho_prior if MAP else None) # need to select correct col in forward, no active_dim

        # unit-level zero-meaned time trend
        self.unit_t_covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(\
            active_dims=torch.tensor([self.d + 2]),\
            lengthscale_prior=unit_lengthscale_prior if MAP else None),\
            outputscale_prior=unit_outputscale_prior if MAP else None) # + 2 because looking at time

        self.unit_indicator_module = myIndicatorKernel(num_tasks=len(train_x[:,-3].unique())) # need to select correct col in forward, no active_dim

        # drift process for treatment effect
        self.drift_t_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(\
                active_dims=torch.tensor([self.d + 2]),\
                lengthscale_prior=drift_lengthscale_prior if MAP else None),\
                outputscale_prior=drift_outputscale_prior if MAP else None) # + 2 because looking at time # need to select correct col in forward, no active_dim
        self.drift_indicator_module = DriftIndicatorKernel(num_tasks=self.num_groups)
        


    def forward(self, x):
        if len(x.shape)==2:
            group = x[:,-2].reshape((-1,1)).long()
            units = x[:,-3].reshape((-1,1)).long()
            ts = x[:,-1]
        else:
            group = x[0,:,-2].reshape((-1,1)).long()
            units = x[0,:,-3].reshape((-1,1)).long()
            ts = x[0,:,-1]

        # only non-zero unit-level mean
        # mu = self.unit_mean_module(x)
        mu = self.group_mean_module(group) + self.unit_mean_module(units) 
        mu = mu.reshape(-1,)
        
        # covariance for time trends
        covar_group_t = self.group_t_covar_module(x)
        covar_group_index = self.group_index_module(group)
        covar_unit_t = self.unit_t_covar_module(x)
        covar_unit_indicator = self.unit_indicator_module(units)
        covar = covar_group_t.mul(covar_group_index) + covar_unit_t.mul(covar_unit_indicator) # multiplied bc of .mul

        #if self.drift_t_module.T0 is not None: aka prior on treatment
        covar_drift_indicator = self.drift_indicator_module((group==1) & (ts>=self.T0).reshape(-1,1))
        covar_drift_t = self.drift_t_module(x)
        covar += covar_drift_t.mul(covar_drift_indicator)
        

        # marginalize covariates effects
        # Marginalizing out the continuous covariates (approximation)
        x_covar_module = self.x_covar_module(x)
        covar += x_covar_module

        return gpytorch.distributions.MultivariateNormal(mu.double(), covar.double())
