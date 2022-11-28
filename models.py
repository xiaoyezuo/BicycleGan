import torchvision
from torchvision.models import resnet18
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import pdb
from datasets import *
##############################
#        Encoder 
##############################
# https://medium.com/@sahil_g/bicyclegan-c104b2c22448
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        """ The encoder used in both cVAE-GAN and cLR-GAN, which encode image B or B_hat to latent vector
            This encoder uses resnet-18 to extract features, and further encode them into a distribution
            similar to VAE encoder. 

            Note: You may either add "reparametrization trick" and "KL divergence" or in the train.py file
            
            Args in constructor: 
                latent_dim: latent dimension for z 
  
            Args in forward function: 
                img: image input (from domain B)

            Returns: 
                mu: mean of the latent code 
                logvar: sigma of the latent code 
        """

        # Extracts features at the last fully-connected
        resnet18_model = resnet18(pretrained=True)      
        self.feature_extractor = nn.Sequential(*list(resnet18_model.children())[:-3])
        self.pooling = nn.AvgPool2d(kernel_size=8, stride=8, padding=0)

        # Output is mu and log(var) for reparameterization trick used in VAEs
        self.fc_mu = nn.Linear(256, latent_dim)
        self.fc_logvar = nn.Linear(256, latent_dim)

    def forward(self, img):
        out = self.feature_extractor(img)
        out = self.pooling(out)
        out = out.view(out.size(0), -1)
        mu = self.fc_mu(out)#mean
        logvar = self.fc_logvar(out)#standard deviation
        return mu, logvar

#sample latent vector from encoded distribution
def latent_sample(mu, logvar):
    eps = torch.randn(mu.shape[0], mu.shape[1])
    z = mu + torch.exp(logvar)*eps
    return z

##############################
#        Generator 
##############################
class Generator(nn.Module):
    """ The generator used in both cVAE-GAN and cLR-GAN, which transform A to B
        
        Args in constructor: 
            latent_dim: latent dimension for z 
            image_shape: (channel, h, w), you may need this to specify the output dimension (optional)
        
        Args in forward function: 
            x: image input (from domain A)
            z: latent vector (encoded B)

        Returns: 
            fake_B: generated image in domain B
    """
    def __init__(self, latent_dim, img_shape):
        super(Generator, self).__init__()
        channels, self.h, self.w = img_shape
        self.unet = UNet(latent_dim)

    def forward(self, x, z):
        x = torch.cat((x,z), dim=1)
        out = self.unet(x)
        return out

##############################
#        Discriminator
##############################
class Discriminator(nn.Module):
    def __init__(self, input_shape):
        super(Discriminator, self).__init__()
        """ The discriminator used in both cVAE-GAN and cLR-GAN
            
            Args in constructor: 
                in_channels: number of channel in image (default: 3 for RGB)

            Args in forward function: 
                x: image input (real_B, fake_B)
 
            Returns: 
                discriminator output: could be a single value or a matrix depending on the type of GAN
        """
        def discriminator_block(in_filters, out_filters, normalize=True):
            """Returns downsampling layers of each discriminator block"""
            layers = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_filters, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers        
        
        channels, _, _ = input_shape
        # Extracts discriminator models
        self.models = nn.ModuleList()
        for i in range(3):
            self.models.add_module(
                "disc_%d" % i,
                nn.Sequential(
                    *discriminator_block(channels, 64, normalize=False),
                    *discriminator_block(64, 128),
                    *discriminator_block(128, 256),
                    *discriminator_block(256, 512),
                    nn.Conv2d(512, 1, 3, padding=1)
                ),
            )        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, x):
        outputs = []
        for m in self.models:
            outputs.append(m(x))
            x = self.downsample(x)
        return outputs
    
    def compute_loss(self, x, gt):
        """Computes the MSE between model output and scalar gt"""
        loss = sum([torch.mean((out - gt) ** 2) for out in self.forward(x)])
        return loss


#UNet
class UNet(nn.Module):
    def __init__(self, latent_dim):
        super().__init__()
        #encoders
        self.e1 = encoder(3+latent_dim, 64)
        self.e2 = encoder(64, 128)
        self.e3 = encoder(128, 256)
        self.e4 = encoder(256, 512)

        #bottleneck
        self.b = conv_block(512, 1024)

        #decoders
        self.d1 = decoder(1024,512)
        self.d2 = decoder(512,256)
        self.d3 = decoder(256,128)
        self.d4 = decoder(128,64)

        #classifier
        self.head = nn.Conv2d(64, 3, kernel_size=1)

    def forward(self, x):
        #encoder
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)
        #bottleneck
        b = self.b(p4)
        #decoder
        d1 = self.d1(b, s4)
        d2 = self.d2(d1,s3)
        d3 = self.d3(d2,s2)
        d4 = self.d4(d3,s1)
        #classifier
        out = self.head(d4)
        return out

#UNet block
class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels) #helps the network to reduce internal covariance shift and make it more stable while training
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.relu(x)
        x = self.bn2(self.conv2(x))
        x = self.relu(x)
        return x
    
#UNet encoder
class encoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.conv = conv_block(in_channel, out_channel)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.conv(x)
        p = self.pool(x)
        return x,p

#UNet decoder
class decoder(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.upconvs = nn.ConvTranspose2d(in_channel, out_channel, kernel_size=2, stride=2)
        self.conv = conv_block(out_channel*2, out_channel)

    def forward(self, x, skip):
        x = self.upconvs(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x
    
    def crop(self, enc_ft, x):
        _,_,H,W = x.shape
        enc_ft = torchvision.transforms.CenterCrop([H,W])(enc_ft)
        return enc_ft

#test encoder
# encoder = Encoder(8)
# #test unet 
# z = torch.randn(1)
# gen = Generator(8, (3, 128, 128))
# x = torch.randn(1,3,128,128)
# output = gen(x, z)
# print(output.shape)