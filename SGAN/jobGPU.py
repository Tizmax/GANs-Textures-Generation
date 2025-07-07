import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from SGAN import Generator, Discriminator
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--texturePath', required=True, help='path to texture image folder')
parser.add_argument('--latentCanal', type=int, default=128, help='number of canal for the latent')
parser.add_argument('--latentSize', type=int, default=4, help='height/width of the latent')
parser.add_argument('--patchSize', type=int, default=64, help='height/width of a patch')
parser.add_argument('--batchSize', type=int, default=16, help='number of patch to extract in a single batch')
parser.add_argument('--sampleLatentSize', type=int, default=16, help='height/width of the latent in order to generate a sample')

opt = parser.parse_args()

LATENT_C = opt.latentCanal
Z_H, Z_W = opt.latentSize, opt.latentSize
PATCH_SIZE = opt.patchSize
BATCH_SIZE = opt.batchSize
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======== UTILS ========
def sample_z(batch_size):
    return torch.randn(batch_size, LATENT_C, Z_H, Z_W, device=DEVICE)


transform = transforms.Compose([
    transforms.RandomCrop(PATCH_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # [0,1] → [-1,1]
])

real_img = Image.open(opt.texturePath).convert("RGB")


# ======== INIT ========
G = Generator(LATENT_C).to(DEVICE)
D = Discriminator().to(DEVICE)
opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

# ======== ENTRAÎNEMENT ========

real_label = 1
fake_label = 0
loss = nn.BCELoss()

# === Data ===

dis_losses = []
gen_losses = []
real_scores = []
fake_scores = []

for epoch in range(5001):
    z = sample_z(BATCH_SIZE)
    fake_img = G(z)

    # === Discriminateur ===
    real_patch = torch.cat([transform(real_img).unsqueeze(0).to(DEVICE) for _ in range(BATCH_SIZE)],dim=0)

    real_score = D(real_patch)
    fake_score = D(fake_img.detach())

    loss_D_real = loss(real_score, torch.zeros_like(real_score)+real_label)
    loss_D_fake = loss(fake_score, torch.zeros_like(fake_score)+fake_label)

    loss_D = loss_D_real + loss_D_fake
    
    opt_D.zero_grad()
    loss_D.backward()
    opt_D.step()

    # === Générateur ===
    
    score = D(fake_img)
    loss_G = loss(score, torch.zeros_like(score)+real_label)

    opt_G.zero_grad()
    loss_G.backward()
    opt_G.step()

    # === Visualisation ===

    dis_losses.append(loss_D.cpu().detach().numpy())
    gen_losses.append(loss_G.cpu().detach().numpy())
    real_scores.append(torch.mean(real_score).cpu().detach().numpy())
    fake_scores.append(torch.mean(fake_score).cpu().detach().numpy())

    # === Visualisation ===
    if epoch % 500 == 0:
        print(f"[{epoch}] Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")
        with torch.no_grad():
            test_z = sample_z(1)
            gen = G(test_z).squeeze().permute(1, 2, 0).cpu().numpy()
            gen = (gen + 1) / 2  # [-1,1] → [0,1]
            plt.figure()
            plt.imshow(gen)
            plt.axis("off")
            plt.savefig(f'../Textures/Generated/SGAN/gravelE{epoch}GPU.png')

plt.figure()
test_z = torch.randn(1, LATENT_C, opt.sampleLatentSize, opt.sampleLatentSize, device=DEVICE)
gen = G(test_z).squeeze().permute(1, 2, 0).cpu().detach().numpy()
gen = (gen + 1) / 2  # [-1,1] → [0,1]
plt.imshow(gen)
plt.axis("off")
plt.savefig('../Textures/Generated/SGAN/gravelSampleGPU.png')

plt.figure()
plt.plot(dis_losses,label='Discriminator losses')
plt.plot(gen_losses,label='Generator losses')
plt.legend()
plt.xlabel('Epochs')
plt.savefig('../Textures/Generated/SGAN/lossesGPU.png')

plt.figure()
plt.plot(real_scores,label='real scores')
plt.plot(fake_scores,label='fake scores')
plt.legend()
plt.xlabel('Epochs')
plt.savefig('../Textures/Generated/SGAN/scoresGPU.png')