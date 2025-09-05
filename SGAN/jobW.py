import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from SGAN import Generator, Discriminator
import argparse
import hashlib
import json
import os

parser = argparse.ArgumentParser()

# Textures args
parser.add_argument('--textureName', required=True, help='path to texture image folder')
parser.add_argument('--originalPath', type=str, default="../Textures/Original/", help='path to texture image folder')
parser.add_argument('--generatedPath', type=str, default="../Textures/Generated/SGAN/", help='path to texture image folder')

# Latent args
parser.add_argument('--latentCanal', type=int, default=50, help='number of canal for the latent')
parser.add_argument('--latentSize', type=int, default=4, help='height/width of the latent')
parser.add_argument('--sampleLatentSize', type=int, default=16, help='height/width of the latent in order to generate a sample')

# Nets args
parser.add_argument('--netDepth', type=int, default=5, help='4|5|6 - number of convolutionals layers')
parser.add_argument('--lrG', type=float, default=1e-3, help='learning rate of the Generator\'s optimizer')
parser.add_argument('--lrD', type=float, default=1e-4, help='learning rate of the Discriminator\'s optimizer')

# Training args
parser.add_argument('--batchSize', type=int, default=16, help='number of patch to extract in a single batch')
parser.add_argument('--epoch', type=int, default=5001, help='number of epochs') 
parser.add_argument('--multD', type=int, default=5, help='number of Discriminator\'s training loop for 1 Generator\'s loop')

# Job args
parser.add_argument('--prefix', type=str , default='', help='prefix to add to the job\'s name')
parser.add_argument('--suffix', type=str , default='', help='suffix to add to the job\'s name')

# Reg args
parser.add_argument('--weightDecay', type=float , default=1e-3, help='weight decay to add to AdamW regularizer')
parser.add_argument('--plotRegLosses', type=bool , default=False, help='Plot the impact of the regularization term on the loss')

opt = parser.parse_args()

if opt.netDepth not in {4,5,6}:
    raise Exception("netDepth should be in {4,5,6}") 



# Hash des parametres
params_dict = vars(opt)
PREFIX = params_dict.pop("prefix")
SUFFIX = params_dict.pop("suffix")
params_json = json.dumps(params_dict, sort_keys=True)
params_hash = hashlib.sha256(params_json.encode('utf-8')).hexdigest()[:12] 
print(f"folder : {params_hash}")

LATENT_C = opt.latentCanal
Z_H, Z_W = opt.latentSize, opt.latentSize
PATCH_SIZE = opt.latentSize*(2**opt.netDepth) ## patch size depends directly of the latent spatial size
BATCH_SIZE = opt.batchSize
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_NAME = opt.textureName.split(".")[0]
OUTPUT_DIR = f"{opt.generatedPath}{IMG_NAME}/{PREFIX}{params_hash}{SUFFIX}"


os.makedirs(OUTPUT_DIR, exist_ok=True)

# Save parameters
with open(os.path.join(OUTPUT_DIR, 'params.json'), 'w') as f:
    json.dump(params_dict, f, indent=4, sort_keys=True)


# ======== UTILS ========
def sample_z(batch_size):
    return torch.randn(batch_size, LATENT_C, Z_H, Z_W, device=DEVICE)


transform = transforms.Compose([
    transforms.RandomCrop(PATCH_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)  # [0,1] → [-1,1]
])

real_img = Image.open(opt.originalPath + opt.textureName).convert("RGB")


# ======== INIT ========
ndf = [64, 128, 256, 512, 1024]
ndf = ndf[:opt.netDepth-1]
ngf = ndf[::-1]

G = Generator(LATENT_C, ngf=ngf).to(DEVICE)
D = Discriminator(ndf=ndf).to(DEVICE)
opt_G = optim.AdamW(G.parameters(), lr=opt.lrG, betas=(0.5, 0.999), weight_decay=opt.weightDecay)
opt_D = optim.Adam(D.parameters(), lr=opt.lrD, betas=(0.5, 0.999))

# ======== ENTRAÎNEMENT ========

real_label = 1
fake_label = 0
loss = nn.BCELoss()

# === Data ===

dis_losses = []
gen_losses = []
real_scores = []
fake_scores = []
reg_losses_G = []
reg_losses_D = []

for epoch in range(opt.epoch):

    for i in range(opt.multD):
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

    gen_losses.append(float(loss_G.cpu().detach().numpy()))
    real_scores.append(torch.mean(real_score).cpu().detach().numpy())
    fake_scores.append(torch.mean(fake_score).cpu().detach().numpy())

    # === Visualisation ===
    with torch.no_grad():
        
        real_score_plot = D(real_patch)
        fake_score_plot = D(fake_img) 
        
        loss_D_real_plot = loss(real_score_plot, torch.zeros_like(real_score_plot)+real_label)
        loss_D_fake_plot = loss(fake_score_plot, torch.zeros_like(fake_score_plot)+fake_label)
        
        loss_D_plot = loss_D_real_plot + loss_D_fake_plot
        dis_losses.append(loss_D_plot.item())
        
        #score = D(fake_img)
        loss_G = loss(fake_score_plot, torch.zeros_like(fake_score_plot)+real_label)

        if opt.plotRegLosses:
            reg_loss_G = 0
            reg_loss_D = 0
            for param in G.parameters():
                norm2 = param.view(-1).norm()**2
                reg_loss_G += norm2.item()
            for param in D.parameters():
                norm2 = param.view(-1).norm()**2
                reg_loss_D += norm2.item()
            reg_losses_G.append(reg_loss_G)
            reg_losses_D.append(reg_loss_D)
            
        if epoch % 500 == 0:
            test_z = sample_z(1)
            gen = G(test_z).squeeze().permute(1, 2, 0).cpu().numpy()
            gen = (gen + 1) / 2  # [-1,1] → [0,1]
            plt.imsave(f'{OUTPUT_DIR}/E{epoch}.png', gen, vmin=0, vmax=1)

torch.save(G.state_dict(), f"{OUTPUT_DIR}/net_G.pth")
torch.save(D.state_dict(), f"{OUTPUT_DIR}/net_D.pth")

test_z = torch.randn(1, LATENT_C, opt.sampleLatentSize, opt.sampleLatentSize, device=DEVICE)
gen = G(test_z).squeeze().permute(1, 2, 0).cpu().detach().numpy()
gen = (gen + 1) / 2  # [-1,1] → [0,1]
plt.imsave(f'{OUTPUT_DIR}/Sample.png', gen, vmin=0, vmax=1)


# Save losses
with open(os.path.join(OUTPUT_DIR, 'losses.json'), 'w') as f:
    json.dump({'dis':dis_losses, 'gen':gen_losses}, f, indent=4)
    
plt.figure()
list_epoch = range(1,opt.epoch+1)
plt.plot(list_epoch, dis_losses,label=r'$\mathcal{L}_D(\phi^{t}, \theta^{t})$')
plt.plot(list_epoch, gen_losses,label=r'$\mathcal{L}_G(\phi^{t}, \theta^{t})$')
plt.legend()
plt.xlabel('Epochs')
plt.savefig(f'{OUTPUT_DIR}/Losses.png')    

if opt.plotRegLosses:
    plt.figure()
    plt.plot(list_epoch, reg_losses_G,label=r'$\|\phi\|^2$')
    plt.legend()
    plt.xlabel('Epochs')
    plt.savefig(f'{OUTPUT_DIR}/regLossesG.png')
    
    plt.figure()
    plt.plot(list_epoch, reg_losses_D,label=r'$\|\theta\|^2$')
    plt.legend()
    plt.xlabel('Epochs')
    plt.savefig(f'{OUTPUT_DIR}/regLossesD.png')

plt.figure()
plt.plot(list_epoch, real_scores,label='real scores')
plt.plot(list_epoch, fake_scores,label='fake scores')
plt.legend()
plt.xlabel('Epochs')
plt.savefig(f'{OUTPUT_DIR}/Scores.png')