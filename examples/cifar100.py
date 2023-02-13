
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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


train_loader = torch.utils.data.DataLoader(datasets.CIFAR100('./data', train=True, download=True,
    transform=transforms.ToTensor()), batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(datasets.CIFAR100('./data', train=False, download=True,
    transform=transforms.ToTensor()), batch_size=64)



class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x

class BasicBlockEnc(nn.Module):
    

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = in_planes*stride

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        if stride == 1:
            self.shortcut = nn.Sequential()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes/stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)
        # self.bn1 could have been placed here, but that messes up the order of the layers when printing the class

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn2(self.conv2(x)))
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNet18Enc(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, nc=3):
        super().__init__()
        self.in_planes = 64
        self.z_dim = z_dim
        self.conv1 = nn.Conv2d(nc, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlockEnc, 64, num_Blocks[0], stride=1)
        self.layer2 = self._make_layer(BasicBlockEnc, 128, num_Blocks[1], stride=2)
        self.layer3 = self._make_layer(BasicBlockEnc, 256, num_Blocks[2], stride=2)
        self.layer4 = self._make_layer(BasicBlockEnc, 512, num_Blocks[3], stride=2)
        self.linear = nn.Linear(512, 2 * z_dim)

    def _make_layer(self, BasicBlockEnc, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in strides:
            layers += [BasicBlockEnc(self.in_planes, stride)]
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        x = torch.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        mu = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]
        return mu, logvar

class ResNet18Dec(nn.Module):

    def __init__(self, num_Blocks=[2,2,2,2], z_dim=10, nc=3):
        super().__init__()
        self.in_planes = 512

        self.linear = nn.Linear(z_dim, 512)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=1)

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1]*(num_Blocks-1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=4)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = torch.sigmoid(self.conv1(x))
        x = x.view(x.size(0), 3, 32, 32)
        return x

class ModelVAE(torch.nn.Module):
    
    def __init__(self, h_dim, z_dim, activation=F.relu, distribution='normal'):
        """
        ModelVAE initializer
        :param h_dim: dimension of the hidden layers
        :param z_dim: dimension of the latent representation
        :param activation: callable activation function
        :param distribution: string either `normal` or `vmf`, indicates which distribution to use
        """
        super(ModelVAE, self).__init__()
        
        self.z_dim, self.activation, self.distribution = z_dim, activation, distribution
        self.encoder = ResNet18Enc(z_dim = z_dim)
        self.decoder = ResNet18Dec(z_dim = z_dim)
        # 2 hidden layers encoder
        self.fc_e0 = nn.Linear(3072, h_dim * 2)
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
        self.fc_logits = nn.Linear(h_dim * 2, 3072)

    def encode(self, x):
        # 2 hidden layers encoder
        z_mean, z_var = self.encoder(x)
        z_var = z_var.exp()

        if self.distribution == 'normal' or self.distribution == 'binary'  :
            # compute mean and std of the normal distribution
            z_mean = z_mean
            z_var = F.softplus(z_var)
        elif self.distribution == 'vmf':
            # compute mean and concentration of the von Mises-Fisher
            z_mean = z_mean
            z_mean = z_mean / z_mean.norm(dim=-1, keepdim=True)
            # the `+ 1` prevent collapsing behaviors
            z_var = F.softplus(z_var) + 1
        else:
            raise NotImplemented
        
        return z_mean, z_var
        
    def decode(self, z):
        x = self.decoder(z)
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

    z_mean, z_var = model.encode(x)
    q_z, p_z = model.reparameterize(z_mean, z_var)
    z = q_z.rsample(torch.Size([n]))
    x_mb_ = model.decode(z)

    log_p_z = p_z.log_prob(z)

    if model.distribution == 'normal' or model.distribution == 'binary':
        log_p_z = log_p_z.sum(-1)

    log_p_x_z = -nn.BCEWithLogitsLoss(reduction='sum')(x_mb_, x)

    log_q_z_x = q_z.log_prob(z)

    if model.distribution == 'normal' or model.distribution == 'binary':
        log_q_z_x = log_q_z_x.sum(-1)

    return ((log_p_x_z + log_p_z - log_q_z_x).t().logsumexp(-1) - np.log(n)).mean()


def train(model, optimizer):
    for i, (x_mb, y_mb) in enumerate(train_loader):

            optimizer.zero_grad()
            
            x_mb = x_mb.to(device)
            (z_mean,z_var), (q_z, p_z), _, x_mb_ = model(x_mb)

            loss_recon = nn.BCEWithLogitsLoss(reduction='sum')(x_mb_, x_mb)

            if model.distribution == 'normal':
                loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean()
            elif model.distribution == 'vmf':
                loss_KL = torch.distributions.kl.kl_divergence(q_z, p_z).mean()
            elif model.distribution == 'binary':
                loss_KL = -0.5 * torch.mean(1 + torch.log(z_var) - (abs(z_mean)-1).pow(2) - z_var)
            else:
                raise NotImplemented

            loss = loss_recon + loss_KL

            loss.backward()
            optimizer.step()
            
            
def test(model, optimizer):
    print_ = defaultdict(list)
    
    for i, (x_mb, y_mb) in enumerate(train_loader):
        (z_mean, z_var),_,_,_ = model(x_mb)
        if i == 0:
          z_means = z_mean.detach().cpu().numpy()
        else:
          z_means = np.concatenate((z_means, z_mean.detach().cpu().numpy()))
        print(z_means.shape)
        print(z_mean.detach().cpu().numpy())
        print(z_means)


    gm = GaussianMixture(n_components=100, random_state=0).fit(z_means)
    
    for x_mb, y_mb in test_loader:
    
        # dynamic binarization
        x_mb = (x_mb > torch.distributions.Uniform(0, 1).sample(x_mb.shape)).float()
        x_mb = x_mb.to(device)
        (z_mean, z_var), (q_z, p_z), _, x_mb_ = model(x_mb_)
        y_gm = gm.predict(z_mean.detach().cpu().numpy())
        NMI = normalized_mutual_info_score(y_mb.detach().cpu().numpy(), y_gm)
        print_['NMI'].append(NMI)

        print_['recon loss'].append(float(nn.BCEWithLogitsLoss(reduction='sum')(x_mb_,
            x_mb).data))
        
        if model.distribution == 'normal':
            print_['KL'].append(float(torch.distributions.kl.kl_divergence(q_z, p_z).sum(-1).mean().data))
        elif model.distribution == 'vmf':
            print_['KL'].append(float(torch.distributions.kl.kl_divergence(q_z, p_z).mean().data))
        elif model.distribution == 'binary':
            print_['KL'].append(float((-0.5 * torch.mean(1 + torch.log(z_var) - (abs(z_mean)-2).pow(2) - z_var)).data))
        else:
            raise NotImplemented
        
        print_['ELBO'].append(- print_['recon loss'][-1] - print_['KL'][-1])
        print_['LL'].append(float(log_likelihood(model, x_mb).data))
    
    print({k: np.mean(v) for k, v in print_.items()})


# hidden dimension and dimension of latent space
H_DIM = 128
EPOCHS = 20

for Z_DIM in [8, 16, 32, 64, 128]:
    print("z_dim: ", Z_DIM)
    # normal VAE
    modelN = ModelVAE(h_dim=H_DIM, z_dim=Z_DIM, distribution='normal')
    modelN = modelN.to(device)
    optimizerN = optim.Adam(modelN.parameters(), lr=1e-3)

    # hyper-spherical  VAE
    modelS = ModelVAE(h_dim=H_DIM, z_dim=Z_DIM + 1, distribution='vmf')
    modelS = modelS.to(device)
    optimizerS = optim.Adam(modelS.parameters(), lr=1e-3)

    # binary  VAE
    modelB = ModelVAE(h_dim=H_DIM, z_dim=Z_DIM, distribution='binary')
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

    print('##### Normal VAE #####')

    for epoch in range(EPOCHS):
        # training for 1 epoch
        start = time.process_time()
        train(modelN, optimizerN)
        print("training time: ",time.process_time() - start)

        # test
        test(modelN, optimizerN)

        print()

    print('##### Hyper-spherical VAE #####')

    for epoch in range(EPOCHS):
        # training for 1 epoch
        start = time.process_time()

        train(modelS, optimizerS)

        print("training time: ",time.process_time() - start)
        # test
        test(modelS, optimizerS)
        print()
    

   