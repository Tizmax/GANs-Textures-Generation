import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from SGAN import Generator, Discriminator, random_patch_batch

LATENT_C = 128
Z_H, Z_W = 32, 32
UPSCALE = 8
IMG_SIZE = Z_H * UPSCALE  # 128
PATCH_SIZE = 64
BATCH_SIZE = 16
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ======== UTILS ========
def sample_z(batch_size):
    return torch.randn(batch_size, LATENT_C, Z_H, Z_W, device=DEVICE)


transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.CenterCrop(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # [0,1] → [-1,1]
])

img = Image.open("../Textures/Original/gravel.jpg").convert("RGB")
real_img = transform(img).unsqueeze(0).to(DEVICE)  # (1, 3, H, W)

plt.figure()
plt.imshow((real_img.squeeze().permute(1, 2, 0).cpu().numpy() + 1)/2)
plt.savefig('../Textures/Generated/SGAN/gravelSource.png')

# ======== INIT ========
G = Generator(LATENT_C).to(DEVICE)
D = Discriminator().to(DEVICE)
opt_G = optim.Adam(G.parameters(), lr=2e-4, betas=(0.5, 0.999))
opt_D = optim.Adam(D.parameters(), lr=2e-4, betas=(0.5, 0.999))

# === Data ===

dis_losses = []
gen_losses = []
real_scores = []
fake_scores = []
# ======== ENTRAÎNEMENT ========

real_label = 1
fake_label = 0
loss = nn.BCELoss()

for epoch in range(5001):
    z = sample_z(BATCH_SIZE)
    fake_img = G(z)

    # === Discriminateur ===
    real_patch = random_patch_batch(real_img, PATCH_SIZE, BATCH_SIZE)
    fake_patch = random_patch_batch(fake_img.detach(), PATCH_SIZE, BATCH_SIZE)

    real_score = D(real_patch)
    fake_score = D(fake_patch)

    loss_D_real = loss(real_score, torch.zeros_like(real_score)+real_label)
    loss_D_fake = loss(fake_score, torch.zeros_like(fake_score)+fake_label)

    loss_D = loss_D_real + loss_D_fake
    
    opt_D.zero_grad()
    loss_D.backward()
    opt_D.step()

    # === Générateur ===
    fake_patch = random_patch_batch(fake_img, PATCH_SIZE, BATCH_SIZE)
    
    score = D(fake_patch)
    loss_G = loss(score, torch.zeros_like(score)+real_label)

    opt_G.zero_grad()
    loss_G.backward()
    opt_G.step()

    # === Visualisation ===

    dis_losses.append(loss_D.detach().numpy())
    gen_losses.append(loss_G.detach().numpy())
    real_scores.append(torch.mean(real_score).detach().numpy())
    fake_scores.append(torch.mean(fake_score).detach().numpy())
    
    
    if epoch % 500 == 0:
        print(f"[{epoch}] Loss D: {loss_D.item():.4f}, Loss G: {loss_G.item():.4f}")
        with torch.no_grad():
            test_z = sample_z(1)
            gen = G(test_z).squeeze().permute(1, 2, 0).cpu().numpy()
            gen = (gen + 1) / 2  # [-1,1] → [0,1]
            plt.imshow(gen)
            plt.axis("off")
            plt.show()



plt.figure()
test_z = sample_z(1)
gen = G(test_z).squeeze().permute(1, 2, 0).cpu().detach().numpy()
gen = (gen + 1) / 2  # [-1,1] → [0,1]
plt.imshow(gen)
plt.axis("off")
plt.savefig('../Textures/Generated/SGAN/gravelSample.png')

plt.figure()
plt.plot(dis_losses,label='Discriminator losses')
plt.plot(gen_losses,label='Generator losses')
plt.legend()
plt.xlabel('Epochs')
plt.savefig('../Textures/Generated/SGAN/losses.png')

plt.figure()
plt.plot(real_scores,label='real scores')
plt.plot(fake_scores,label='fake scores')
plt.legend()
plt.xlabel('Epochs')
plt.savefig('../Textures/Generated/SGAN/scores.png')