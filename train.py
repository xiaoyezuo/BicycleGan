import warnings
warnings.filterwarnings("ignore")
from torch.utils import data
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from vis_tools import *
from datasets import *
from models import *
import argparse, os
import itertools
import torch
import time
import pdb
import datetime
import sys

# Training Configurations 
# (You may put your needed configuration here. Please feel free to add more or use argparse. )
img_dir = '/Users/xiaoyezuo/Desktop/bicyclegan/edges2shoes/train/'
img_shape = (3, 128, 128) # Please use this image dimension faster training purpose

epoch = 0
num_epochs = 20 
batch_size = 32
lr_rate =  1e-4 	      # Adam optimizer learning rate
betas1 = 0.5			  # Adam optimizer beta 1, beta 2
betas2 = 0.999
lambda_pixel = 10       # Loss weights for pixel loss
lambda_latent =  0.5     # Loss weights for latent regression 
lambda_kl =  0.01         # Loss weights for kl divergence
latent_dim = 8         # latent dimension for the encoded images from domain B
gpu_id = 0

sample_interval = 400          #interval between saving generator samples
checkpoint_interval = -1       #interval between model checkpoints


cuda = True if torch.cuda.is_available() else False   #availability of GPU
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# Normalize image tensor
def norm(image):
	return (image/255.0-0.5)*2.0

# Denormalize image tensor
def denorm(tensor):
	return ((tensor+1.0)/2.0)*255.0

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)



# Reparameterization helper function 
# (You may need this helper function here or inside models.py, depending on your encoder implementation)
def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), latent_dim))))
    z = sampled_z * std + mu
    return z


# Random seeds (optional)
torch.manual_seed(1); np.random.seed(1)

# Define DataLoader
dataset = Edge2Shoe(img_dir)
loader = data.DataLoader(dataset, batch_size=batch_size)

# Loss functions
mae_loss = torch.nn.L1Loss().to(gpu_id)

# Define generator, encoder and discriminators
generator = Generator(latent_dim, img_shape).to(gpu_id)
encoder = Encoder(latent_dim).to(gpu_id)
D_VAE = Discriminator().to(gpu_id)
D_LR = Discriminator().to(gpu_id)


if cuda:
    generator = generator.cuda()
    encoder.cuda()
    D_VAE = D_VAE.cuda()
    D_LR = D_LR.cuda()
    mae_loss.cuda()    # Initialize weights
    generator.apply(weights_init_normal)
    D_VAE.apply(weights_init_normal)
    D_LR.apply(weights_init_normal)


# Define optimizers for networks
optimizer_E = torch.optim.Adam(encoder.parameters(), lr=lr_rate, betas=betas)
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr_rate, betas=betas)
optimizer_D_VAE = torch.optim.Adam(D_VAE.parameters(), lr=lr_rate, betas=betas)
optimizer_D_LR = torch.optim.Adam(D_LR.parameters(), lr=lr_rate, betas=betas)

# For adversarial loss (optional to use)
valid = 1
fake = 0

dataset_name = "edges2shoes"

dataloader = DataLoader(
    Edge2Shoe("../../data/%s" % dataset_name, img_shape),
    batch_size=batch_size,
    shuffle=True,
    num_workers=1,
)
val_dataloader = DataLoader(
    Edge2Shoe("../../data/%s" % dataset_name, img_shape, mode="val"),
    batch_size=8,
    shuffle=True,
    num_workers=1,
)

def sample_images(batches_done):    

	generator.eval()
	imgs = next(iter(val_dataloader))
	img_samples = None
	for img_A, img_B in zip(imgs["A"], imgs["B"]):        # Repeat input image by number of desired columns
		real_A = img_A.view(1, *img_A.shape).repeat(latent_dim, 1, 1, 1)
		real_A = Variable(real_A.type(Tensor))        # Sample latent representations
		sampled_z = Variable(Tensor(np.random.normal(0, 1, (latent_dim, latent_dim))))
        # Generate samples
		fake_B = generator(real_A, sampled_z)
        # Concatenate samples horisontally
		fake_B = torch.cat([x for x in fake_B.data.cpu()], -1)
		img_sample = torch.cat((img_A, fake_B), -1)
		img_sample = img_sample.view(1, *img_sample.shape)
        # Concatenate with previous samples vertically
		img_samples = img_sample if img_samples is None else torch.cat((img_samples, img_sample), -2)
	save_image(img_samples, "images/%s/%s.png" % (dataset_name, batches_done), nrow=8, normalize=True)
	generator.train()

# Training
# total_steps = len(loader)*num_epochs; step = 0
# for e in range(num_epochs):
# 	start = time.time()
# 	for idx, data in enumerate(loader):

# 		########## Process Inputs ##########
# 		edge_tensor, rgb_tensor = data
# 		edge_tensor, rgb_tensor = norm(edge_tensor).to(gpu_id), norm(rgb_tensor).to(gpu_id)
# 		real_A = edge_tensor; real_B = rgb_tensor;
prev_time = time.time()
for epoch in range(num_epochs):
    for i, batch in enumerate(dataloader):        # Set model input
        real_A = Variable(batch["A"].type(Tensor))
        real_B = Variable(batch["B"].type(Tensor))        
		# -------------------------------
        #  Train Generator and Encoder
        # ------------------------------- 
		#       
		# optimizer_E.zero_grad()
        optimizer_G.zero_grad()    

		# ----------
        # cVAE-GAN
        # ---------- 
		# Produce output using encoding of B (cVAE-GAN)
        mu, logvar = encoder(real_B)
        encoded_z = reparameterization(mu, logvar)
        fake_B = generator(real_A, encoded_z)        
		# Pixelwise loss of translated image by VAE
        loss_pixel = mae_loss(fake_B, real_B)
        # Kullback-Leibler divergence of encoded B
        loss_kl = 0.5 * torch.sum(torch.exp(logvar) + mu ** 2 - logvar - 1)
        # Adversarial loss
        loss_VAE_GAN = D_VAE.compute_loss(fake_B, valid)    

		# ---------
        # cLR-GAN
        # ---------
		# Produce output using sampled z (cLR-GAN)
        sampled_z = Variable(Tensor(np.random.normal(0, 1, (real_A.size(0), latent_dim))))
        _fake_B = generator(real_A, sampled_z)
        # cLR Loss: Adversarial loss
        loss_LR_GAN = D_LR.compute_loss(_fake_B, valid)    

		# ----------------------------------
        # Total Loss (Generator + Encoder)
        # ----------------------------------  
        loss_GE = loss_VAE_GAN + loss_LR_GAN + lambda_pixel * loss_pixel + lambda_kl * loss_kl        
        loss_GE.backward(retain_graph=True)
        optimizer_E.step()        
		
		# ---------------------
        # Generator Only Loss
        # ---------------------        
		# Latent L1 loss
        _mu, _ = encoder(_fake_B)
        loss_latent = lambda_latent * mae_loss(_mu, sampled_z)
        loss_latent.backward()
        optimizer_G.step()        
		
		# ----------------------------------
        #  Train Discriminator (cVAE-GAN)
        # ----------------------------------        
		# 
        optimizer_D_VAE.zero_grad()        
        loss_D_VAE = D_VAE.compute_loss(real_B, valid) + D_VAE.compute_loss(fake_B.detach(), fake)        
        loss_D_VAE.backward()
        optimizer_D_VAE.step()        
		
		# ---------------------------------
        #  Train Discriminator (cLR-GAN)
        # ---------------------------------       
        optimizer_D_LR.zero_grad()        
        loss_D_LR = D_LR.compute_loss(real_B, valid) + D_LR.compute_loss(_fake_B.detach(), fake)        
        loss_D_LR.backward()
        optimizer_D_LR.step()        
		
		# --------------
        #  Log Progress
        # --------------        
		# Determine approximate time left
		
        batches_done = epoch * len(dataloader) + i
        batches_left = num_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()        # Print log
        sys.stdout.write(
            "\r[Epoch %d/%d] [Batch %d/%d] [D VAE_loss: %f, LR_loss: %f] [G loss: %f, pixel: %f, kl: %f, latent: %f] ETA: %s"
            % (
                epoch,
                num_epochs,
                i,
                len(dataloader),
                loss_D_VAE.item(),
                loss_D_LR.item(),
                loss_GE.item(),
                loss_pixel.item(),
                loss_kl.item(),
                loss_latent.item(),
                time_left,
            )
        )
        if batches_done % sample_interval == 0:
            sample_images(batches_done)


""" Optional TODO: 
			1. You may want to visualize results during training for debugging purpose
			2. Save your model every few iterations
"""