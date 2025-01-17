
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
from torchvision import datasets, transforms
from collections import defaultdict
from sklearn.mixture import GaussianMixture
from sklearn.metrics.cluster import normalized_mutual_info_score
from hyperspherical_vae.distributions import VonMisesFisher
from hyperspherical_vae.distributions import HypersphericalUniform
import time


train_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True,
    transform=transforms.ToTensor()), batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.MNIST('./data', train=False, download=True,
    transform=transforms.ToTensor()), batch_size=64)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ModelVAE(torch.nn.Module):
    
    def __init__(self, h_dim, z_dim, activation=F.relu, distribution='normal', r = 0):
        """
        ModelVAE initializer
        :param h_dim: dimension of the hidden layers
        :param z_dim: dimension of the latent representation
        :param activation: callable activation function
        :param distribution: string either `normal` or `vmf`, indicates which distribution to use
        """
        super(ModelVAE, self).__init__()
        
        self.z_dim, self.activation, self.distribution = z_dim, activation, distribution
        self.r = r
        # 2 hidden layers encoder
        self.fc_e0 = nn.Linear(784, h_dim * 2)
        self.fc_e1 = nn.Linear(h_dim * 2, h_dim)

        if self.distribution == 'normal':
            # compute mean and std of the normal distribution
            self.fc_mean = nn.Linear(h_dim, z_dim)
            self.fc_var =  nn.Linear(h_dim, z_dim)
        elif self.distribution == 'vmf':
            # compute mean and concentration of the von Mises-Fisher
            self.fc_mean = nn.Linear(h_dim, z_dim)
            self.fc_var = nn.Linear(h_dim, 1)
        elif self.distribution == 'binary':
            # compute mean and concentration of the von Mises-Fisher
            self.fc_mean = nn.Linear(h_dim, z_dim)
            self.fc_var = nn.Linear(h_dim, z_dim)
        else:
            raise NotImplemented
            
        # 2 hidden layers decoder
        self.fc_d0 = nn.Linear(z_dim, h_dim)
        self.fc_d1 = nn.Linear(h_dim, h_dim * 2)
        self.fc_logits = nn.Linear(h_dim * 2, 784)

    def encode(self, x):
        # 2 hidden layers encoder
        x = self.activation(self.fc_e0(x))
        x = self.activation(self.fc_e1(x))
        
        if self.distribution == 'normal' or self.distribution == 'binary'  :
            # compute mean and std of the normal distribution
            z_mean = self.fc_mean(x)
            z_var = F.softplus(self.fc_var(x))
        elif self.distribution == 'vmf':
            # compute mean and concentration of the von Mises-Fisher
            z_mean = self.fc_mean(x)
            z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
            # the `+ 1` prevent collapsing behaviors
            z_var = F.softplus(self.fc_var(x)) + 1
        else:
            raise NotImplemented
        
        return z_mean, z_var
        
    def decode(self, z):
        
        x = self.activation(self.fc_d0(z))
        x = self.activation(self.fc_d1(x))
        x = self.fc_logits(x)
        
        return x
        
    def reparameterize(self, z_mean, z_var):
        if self.distribution == 'normal' or self.distribution == 'binary':
            q_z = torch.distributions.normal.Normal(z_mean, z_var)
            p_z = torch.distributions.normal.Normal(torch.zeros_like(z_mean), torch.ones_like(z_var))
        elif self.distribution == 'vmf':
            q_z = VonMisesFisher(z_mean, z_var)
            p_z = HypersphericalUniform(self.z_dim - 1, validate_args=False, device = device)
        else:
            raise NotImplemented

        return q_z, p_z
        
    def forward(self, x): 
        z_mean, z_var = self.encode(x)
        q_z, p_z = self.reparameterize(z_mean, z_var)
        z = q_z.rsample()
        x_ = self.decode(z)
        
        return (z_mean, z_var), (q_z, p_z), z, x_
    
    
def log_likelihood(model, x, n=10):
    """
    :param model: model object
    :param optimizer: optimizer object
    :param n: number of MC samples
    :return: MC estimate of log-likelihood
    """

    z_mean, z_var = model.encode(x.reshape(-1, 784))
    q_z, p_z = model.reparameterize(z_mean, z_var)
    z = q_z.rsample(torch.Size([n]))
    x_mb_ = model.decode(z)

    log_p_z = p_z.log_prob(z)

    if model.distribution == 'normal' or model.distribution == 'binary':
        log_p_z = log_p_z.sum(-1)

    log_p_x_z = -nn.BCEWithLogitsLoss(reduction='none')(x_mb_, x.reshape(-1, 784).repeat((n, 1, 1))).sum(-1)

    log_q_z_x = q_z.log_prob(z)

    if model.distribution == 'normal' or model.distribution == 'binary':
        log_q_z_x = log_q_z_x.sum(-1)

    return ((log_p_x_z + log_p_z - log_q_z_x).t().logsumexp(-1) - np.log(n)).mean()


def train(model, optimizer):
    for i, (x_mb, y_mb) in enumerate(train_loader):

            optimizer.zero_grad()
            
            # dynamic binarization
            x_mb = (x_mb > torch.distributions.Uniform(0, 1).sample(x_mb.shape)).float()
            x_mb = x_mb.to(device)
            (z_mean,z_var), (q_z, p_z), _, x_mb_ = model(x_mb.reshape(-1, 784))

            loss_recon = nn.BCEWithLogitsLoss(reduction='none')(x_mb_, x_mb.reshape(-1, 784)).sum(-1).mean()

            if model.distribution == 'normal':
                loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean()
            elif model.distribution == 'vmf':
                loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).mean()
            elif model.distribution == 'binary':
                loss_KL = -0.5 * torch.mean(1 + torch.log(z_var) - (abs(z_mean)-model.r).pow(2) - z_var)
            else:
                raise NotImplemented

            loss = loss_recon + loss_KL

            loss.backward()
            optimizer.step()
            
            
def test(model, optimizer):
    print_ = defaultdict(list)
    for i, (x_mb, y_mb) in enumerate(train_loader):
        x_mb = x_mb.to(device)

        (z_mean, z_var),_,_,_ = model(x_mb.reshape(-1, 784))
        if i == 0:
            z_means = z_mean.detach().cpu().numpy()
        else:
            z_means = np.concatenate((z_means,z_mean.detach().cpu().numpy()))
   
    gm = GaussianMixture(n_components=10, random_state=0).fit(z_means)
    
    for i, (x_mb, y_mb) in enumerate(test_loader):
    
        # dynamic binarization
        x_mb = (x_mb > torch.distributions.Uniform(0, 1).sample(x_mb.shape)).float()
        x_mb = x_mb.to(device)
        (z_mean, z_var), (q_z, p_z), _, x_mb_ = model(x_mb.reshape(-1, 784))
        y_gm = gm.predict(z_mean.detach().cpu().numpy())
        if i == 0:
            y_gm_sofar = y_gm
            y_sofar = y_mb.detach().cpu().numpy()
        else:
            y_gm_sofar = np.concatenate((y_gm, y_gm_sofar))
            y_sofar = np.concatenate((y_mb.detach().cpu().numpy(),y_sofar))

        NMI = normalized_mutual_info_score(y_sofar, y_gm_sofar)
        
        print_['NMI'].append(NMI)

        print_['recon loss'].append(float(nn.BCEWithLogitsLoss(reduction='none')(x_mb_,
            x_mb.reshape(-1, 784)).sum(-1).mean().data))
        
        if model.distribution == 'normal':
            print_['KL'].append(float(torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean().data))
        elif model.distribution == 'vmf':
            print_['KL'].append(float(torch.distributions.kl.kl_divergence(q_z, p_z).mean().data))
        elif model.distribution == 'binary':
            print_['KL'].append(float((-0.5 * torch.mean(1 + torch.log(z_var) - (abs(z_mean)-model.r).pow(2) - z_var)).data))
        else:
            raise NotImplemented
        
        print_['ELBO'].append(- print_['recon loss'][-1] - print_['KL'][-1])
        print_['LL'].append(float(log_likelihood(model, x_mb).data))
    
    print({k: np.mean(v) for k, v in print_.items()})


# hidden dimension and dimension of latent space
H_DIM = 128

EPOCHS = 20

for Z_DIM in [2, 4, 8, 16, 32, 64]:
  for r in [0.01]:
    print("z_dim: ", Z_DIM)
    # # normal VAE
    # modelN = ModelVAE(h_dim=H_DIM, z_dim=Z_DIM, distribution='normal')
    # modelN = modelN.to(device)
    # optimizerN = optim.Adam(modelN.parameters(), lr=1e-3)

    # # hyper-spherical  VAE
    # modelS = ModelVAE(h_dim=H_DIM, z_dim=Z_DIM + 1, distribution='vmf')
    # modelS = modelS.to(device)
    # optimizerS = optim.Adam(modelS.parameters(), lr=1e-3)

    # binary  VAE
    modelB = ModelVAE(h_dim=H_DIM, z_dim=Z_DIM, distribution='binary', r = r)
    modelB = modelB.to(device)
    optimizerB = optim.Adam(modelB.parameters(), lr=1e-3)
   
    print('##### Binary VAE #####')

    for epoch in range(EPOCHS):
        # training for 1 epoch
        start = time.process_time()
        train(modelB, optimizerB)
        print("training time: ",time.process_time() - start)
        # test
        test(modelB, optimizerB)

        print()

    # print('##### Normal VAE #####')

    # for epoch in range(EPOCHS):
    #     # training for 1 epoch
    #     start = time.process_time()
    #     train(modelN, optimizerN)
    #     print("training time: ",time.process_time() - start)

    #     # test
    #     test(modelN, optimizerN)

    #     print()

    # print('##### Hyper-spherical VAE #####')

    # for epoch in range(EPOCHS):
    #     # training for 1 epoch
    #     start = time.process_time()

    #     train(modelS, optimizerS)

    #     print("training time: ",time.process_time() - start)
    #     # test
    #     test(modelS, optimizerS)
    #     print()
    

   