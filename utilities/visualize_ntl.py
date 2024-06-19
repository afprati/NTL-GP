import torch
import gpytorch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import scipy.special as sps 
import seaborn as sns
import copy
import datetime
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from model.multitaskmodel import MultitaskGPModel



results = torch.load("results/ntl_MAP_model_state.pth")

# Define plotting function
def ax_plot(ax, test_t, X, Y, m, lower, upper, LABEL):
    for i in range(2):
          ax[i].plot(1+X[i, :,-1].detach().numpy(), Y[i,:].detach().numpy(),\
          color='grey', alpha=0.8, label=LABEL)
          ax[i].plot(1+test_t.detach().numpy(), m[i].detach().numpy(),\
               'k--', linewidth=1.0, label='Estimated Y(0)')
          ax[i].fill_between(1+test_t.detach().numpy(), lower[i].detach().numpy(),\
               upper[i].detach().numpy(), alpha=0.5)
          ax[i].legend(loc=2)
          ax[i].set_title("{} Unit {}".format(LABEL, i+1))
          
    param_list = ["likelihood.noise_covar.noise_prior", "t_covar_module.outputscale_prior", "t_covar_module.base_kernel.lengthscale_prior"]
    xmax = [1,1,60]
    labels = ["noise", "os","ls"]
    fig, ax = plt.subplots(nrows=2, ncols=2)
    for i in range(3):
         parts = param_list[i].split(".")
         prior = model
         for part in parts:
              prior = getattr(prior, part)
         m = prior.concentration.item()
         s = prior.rate.item()
         x = np.linspace(0, xmax[i], 10000)
         # pdf = (np.exp(-(np.log(x) - m)**2 / (2 * s**2)) / (x * s * np.sqrt(2 * np.pi)))
         pdf = x**(m-1)*np.exp(-x*s)*s**m/sps.gamma(m)
         ax[int(i/2)][int(i%2)].plot(x, pdf, color='r', linewidth=2)
         ax[int(i/2)][int(i%2)].legend([labels[i]+" a: " + str(np.around(m,1)) + " b: " + str(np.around(s,1))])

    x = np.linspace(-1,1,10000)
    pdf = 1/2*np.ones(x.shape)
    ax[1][1].plot(x, pdf, color='r', linewidth=2)
    ax[1][1].legend(["rho"])
    fig.suptitle("Gamma prior")
    plt.savefig("results/gammaprior.png")
    plt.close()
    return 


def visualize_ntl(data, test_x, test_y, test_g, model, likelihood, T0, obs_le, train_condition):
    # Set into eval mode
    model.eval()
    likelihood.eval()
    for i in range(len(model.x_covar_module)):
        model.x_covar_module[i].c2 = torch.tensor(0.0**2)

    with torch.no_grad(), gpytorch.settings.prior_mode(True), gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
        f_pred = model(test_x)

    K_sum = copy.deepcopy(f_pred.covariance_matrix.detach().numpy())
    mu_sum = copy.deepcopy(f_pred.mean.detach().numpy())

    with torch.no_grad(), gpytorch.settings.prior_mode(False), gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
        f_pred = model(test_x)

    result = pd.DataFrame({
         "t":test_x[:,-1],
         "g": test_g,
         "m": f_pred.mean})
    result = result.groupby(['t','g'], as_index=False)[['m']].mean()
    m_1 = result[result.g==1].m.to_numpy()


    with torch.no_grad(), gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
         f_pred = model(test_x)

    # Get lower and upper confidence bounds
    lower, upper = f_pred.confidence_region()

    obs_ids = data.obs_id.unique()
    
    for obs_id in []:
         mask = (data.obs_id==obs_id).to_numpy()
         test_t = test_x[mask, -1]
         idx = np.argsort(test_t)
         test_t = test_t[[idx]]
         lower_i = lower[mask][idx]
         upper_i = upper[mask][idx]
         m_i = f_pred.mean[mask][idx]
         treatment = data[mask].Treated.unique()[0]
         LABEL = "treated" if treatment else "control"
         y_i = test_y[mask][[idx]]

         plt.rcParams["figure.figsize"] = (15,5)
         plt.scatter(1+test_t.detach().numpy(), y_i.detach().numpy(),\
               color='blue', s=4, label=LABEL + " " + str(obs_id))
         plt.plot(1+test_t.detach().numpy(), m_i.detach().numpy(),\
               'k--', linewidth=2.0, label='Estimated Y(0)')
         plt.fill_between(1+test_t.detach().numpy(), lower_i.detach().numpy(),\
               upper_i.detach().numpy(), alpha=0.3)
         plt.legend(loc=2)
         plt.title("{} Unit {}".format(LABEL, obs_id))
         plt.axvline(x=T0, color='red', linewidth=1.0)
         plt.savefig("results/ntl_{}.png".format(obs_id))
         plt.close()

    model.unit_t_covar_module.outputscale = 0
    model.unit_mean_module.constantvector.data.fill_(0.0)

     # Make predictions
    with torch.no_grad(), gpytorch.settings.prior_mode(False), gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
         f_pred = model(test_x)

    # Get lower and upper confidence bounds
    lower, upper = f_pred.confidence_region()
    
    obs_ids = data.obs_id.unique()
    result = pd.DataFrame({
         "t":test_x[:,-1],
         "g": test_g,
         "upper": upper,
         "lower": lower,
         "m": f_pred.mean,
         "y": test_y,
         "Treated": data.Treated})
    result = result.groupby(['t','g'], as_index=False)[['lower','upper','m','y']].mean()
    fill_alpha = [0.1, 0.2]
    mean_color = ["blue", "tomato"]
    y_color = ["purple", "crimson"]
    # plt.rcParams["figure.figsize"] = (15,5)
    for g in []:
         test_t = np.unique(result[result.g==g].t)
         lower_g = result[result.g==g].lower.to_numpy()
         upper_g = result[result.g==g].upper.to_numpy()
         m_g = result[result.g==g].m.to_numpy()
         if g==0:
              m_g_0 = m_g
              v_0 = (upper_g - lower_g)/2
         else:
              m_g_1 = m_g
              v_1 = (upper_g - lower_g)/2
         y_g = result[result.g==g].y.to_numpy()
         LABEL = "UN" if g==1 else "Not UN"
         plt.rcParams["figure.figsize"] = (15,5)
         plt.scatter(x=1+test_t, y=y_g, c=y_color[0], s=4, label=LABEL + " avg")
         plt.plot(1+test_t, m_g, c=mean_color[0], linewidth=2, label=LABEL +' estimated Y(0)')
         plt.fill_between(1+test_t, lower_g, upper_g, color='grey', alpha=fill_alpha[g], label=LABEL + " 95% CI")
         plt.legend(loc=2)
         plt.title("Averaged " + LABEL + " Group Trends ")
         plt.axvline(x=T0, color='red', linewidth=0.5, linestyle="--")
         plt.savefig("results/ntl_MAP_{}.png".format(LABEL))
         plt.close()




 # Make predictions
    with torch.no_grad(), gpytorch.settings.prior_mode(False), gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
         f_pred = model(test_x)

    # Get lower and upper confidence bounds
    lower, upper = f_pred.confidence_region()
    
    obs_ids = data.obs_id.unique()
    result = pd.DataFrame({
         "t":test_x[:,-1],
         "g": test_g,
         "upper": upper,
         "lower": lower,
         "m": f_pred.mean,
         "y": test_y,
         "Treated": data.Treated})
    result = result.groupby(['t','g'], as_index=False)[['lower','upper','m','y']].mean()
    fill_alpha = [0.1, 0.2]
    mean_color = ["blue", "tomato"]
    y_color = ["purple", "crimson"]
    plt.rcParams["figure.figsize"] = (15,5)
    for g in []:
         test_t = np.unique(result[result.g==g].t)
         lower_g = result[result.g==g].lower.to_numpy()
         upper_g = result[result.g==g].upper.to_numpy()
         m_g = result[result.g==g].m.to_numpy()
     #     if g==0:
     #          m_g_0 = m_g
     #          v_0 = (upper_g - lower_g)/2
     #     else:
     #          m_g_1 = m_g
     #          v_1 = (upper_g - lower_g)/2
         y_g = result[result.g==g].y.to_numpy()
         LABEL = "UN" if g==1 else "Not UN"
         # plt.rcParams["figure.figsize"] = (15,5)
         plt.scatter(x=1+test_t, y=y_g, c=y_color[g], s=4, label=LABEL + " avg")
         plt.plot(1+test_t, m_g, c=mean_color[g], linewidth=2, label=LABEL +' estimated Y')
         plt.fill_between(1+test_t, lower_g, upper_g, color='grey', alpha=fill_alpha[g], label=LABEL + " 95% CI")
         plt.legend(loc=2)
         plt.title("Averaged " + LABEL + " Group Trends ")
         plt.axvline(x=T0, color='red', linewidth=0.5, linestyle="--")



    model.group_t_covar_module.outputscale = 0
    model.group_mean_module.constantvector.data.fill_(0.0)

    with torch.no_grad(), gpytorch.settings.prior_mode(True), gpytorch.settings.fast_computations(covar_root_decomposition=False, log_prob=False, solves=False):
        f_pred = model(test_x)

    K_D = copy.deepcopy(f_pred.covariance_matrix.detach().numpy())
    mu_D = copy.deepcopy(f_pred.mean.detach().numpy())

    

     # verify conditioning on sum of two GPs
    g = 1
    mu_p = mu_D+ K_D.dot(np.linalg.inv(K_sum+np.identity(K_sum.shape[0])*likelihood.noise.item())).dot(test_y-mu_sum)
    K_p = K_D - K_D.dot(np.linalg.inv(K_sum+np.identity(K_sum.shape[0])*likelihood.noise.item())).dot(K_D)

    result = pd.DataFrame({
         "t":test_x[:,-1],
         "g": test_g,
         "m": mu_p,
         "s2": K_p.diagonal(),
         "y": test_y})
    result = result.groupby(['t','g'], as_index=False)[['m','y',"s2"]].mean()

    m_g = result[result.g==g].m.to_numpy()
    test_t = np.unique(result[result.g==g].t)
    plt.rcParams["figure.figsize"] = (15,5)
    plt.plot(1+test_t, m_g, c="darkblue", linewidth=2, label='treatment effect')

    std_p = np.sqrt(result[result.g==g].s2.to_numpy())
    lower_g = m_g - 1.96*std_p
    upper_g = m_g + 1.96*std_p
    #plt.plot(1+test_t,m_1-m_0,  c="blue", linestyle="--",linewidth=1, label='estimated Y(1)-Y(0)')
    plt.fill_between(1+test_t, lower_g, upper_g, color='grey', alpha=fill_alpha[1], label="95% CI")
    plt.legend(loc=2)
    plt.title("Treatment Effect Trend ")
    plt.axvline(x=T0, color='red', linewidth=0.5, linestyle="--")
    plt.savefig("results/ntl_MAP_effect.png")
    plt.close()
    # plt.show()