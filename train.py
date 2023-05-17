import argparse
import itertools
import os

from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from models import Generator
from models import Discriminator
from utils import ReplayBuffer
from utils import LambdaLR
# from utils import Logger
from utils import weights_init_normal
from torchvision.utils import save_image
from datasets import ImageDataset
from metrics import calculate_psnr, calculate_ssim, calculate_dice_coefficient
import time
import os.path as osp

root_dir = osp.dirname(osp.abspath(__file__))
if not osp.exists(osp.join(root_dir, "weights")):
    os.mkdir(osp.join(root_dir, "weights"))
weight_dir = osp.join(root_dir, "weights")
if not osp.exists(osp.join(root_dir, "output")):
    os.mkdir(osp.join(root_dir, "output"))
output_dir = osp.join(root_dir, "output")

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/horse2zebra/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=3e-4, help='initial learning rate')
parser.add_argument('--weight_decay', type=float, default=1e-3, help='weight decay')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=4, help='number of cpu threads to use during batch generation')
opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)
netD_A = Discriminator(opt.input_nc)
netD_B = Discriminator(opt.output_nc)

netG_A2B = netG_A2B.cuda()
netG_B2A = netG_B2A.cuda()
netD_A = netD_A.cuda()
netD_B = netD_B.cuda()

netG_A2B.apply(weights_init_normal)
netG_B2A.apply(weights_init_normal)
netD_A.apply(weights_init_normal)
netD_B.apply(weights_init_normal)

# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.AdamW(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999), weight_decay=opt.weight_decay)
optimizer_D_A = torch.optim.AdamW(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999), weight_decay=opt.weight_decay)
optimizer_D_B = torch.optim.AdamW(netD_B.parameters(), lr=opt.lr, betas=(0.5, 0.999), weight_decay=opt.weight_decay)

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor
target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
    ToTensorV2()
])

test_transform = A.Compose([
    ToTensorV2()
])

train_dataset = ImageDataset(transform=train_transform, unaligned=False, mode="train")
test_dataset = ImageDataset(transform=test_transform, unaligned=False, mode="test")
train_dataloader = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=False)

# Loss plot
# logger = Logger(opt.n_epochs, len(train_dataloader))
###################################

def test(G_A2B, G_B2A, D_A, D_B, overall_score, mode="A2B"):
    if mode == "A2B":
        model = G_A2B
    elif mode == "B2A":
        model = G_B2A

    model.eval()
    metrics = {"psnr": 0, "ssim": 0, "dice": 0, "lcnn": 0, "time": 0}
    with torch.no_grad():
        for i, (img, target, name) in enumerate(test_dataloader):
            img_A = img.cuda()
            img_B = target.detach().squeeze().cpu().numpy()
            start_time = time.time()
            fake_B = 0.5*(model(img_A).data + 1.0)
            metrics["time"] += (time.time()-start_time)
            fake_B = fake_B.detach().squeeze().cpu().numpy()
            metrics["psnr"] += calculate_psnr(img_B, fake_B)
            metrics["ssim"] += calculate_ssim(img_B, fake_B)
            metrics["dice"] += calculate_dice_coefficient(img_B, fake_B)
            save_image(torch.tensor(fake_B), osp.join(output_dir, name[0]))
            
    for key in metrics:
        metrics[key] /= len(test_dataset)
    print("metrics:", metrics)
    cur_score = metrics["psnr"]+metrics["ssim"]+metrics["dice"]-0.05*metrics["time"]
    print("current score:", cur_score, " overall_score:", overall_score)
    if cur_score>overall_score:
        overall_score = cur_score
        # Save models checkpoints
        torch.save(G_A2B.state_dict(), osp.join(weight_dir, 'netG_A2B.pth'))
        torch.save(G_B2A.state_dict(), osp.join(weight_dir, 'netG_B2A.pth'))
        torch.save(D_A.state_dict(), osp.join(weight_dir, 'netD_A.pth'))
        torch.save(D_B.state_dict(), osp.join(weight_dir, 'netD_B.pth'))
    return overall_score

###### Training ######
def train():
    overall_score = 0
    for epoch in range(opt.epoch, opt.n_epochs):
        netG_A2B.train()
        netG_B2A.train()
        netD_A.train()
        netD_B.train()
        loss_dict = {"loss_G":0, "loss_G_identity":0, "loss_G_GAN": 0, "loss_G_cycle": 0, "loss_D": 0}
        for i, (img, target, _) in enumerate(train_dataloader):
            # Set model input
            real_A = img.cuda()
            real_B = target.cuda()

            ###### Generators A2B and B2A ######
            optimizer_G.zero_grad()

            # Identity loss
            # G_A2B(B) should equal B if real B is fed
            same_B = netG_A2B(real_B)
            loss_identity_B = criterion_identity(same_B, real_B)*5.0
            # G_B2A(A) should equal A if real A is fed
            same_A = netG_B2A(real_A)
            loss_identity_A = criterion_identity(same_A, real_A)*5.0

            # GAN loss
            fake_B = netG_A2B(real_A)
            pred_fake = netD_B(fake_B)
            loss_GAN_A2B = criterion_GAN(pred_fake, target_real)

            fake_A = netG_B2A(real_B)
            pred_fake = netD_A(fake_A)
            loss_GAN_B2A = criterion_GAN(pred_fake, target_real)

            # Cycle loss
            recovered_A = netG_B2A(fake_B)
            loss_cycle_ABA = criterion_cycle(recovered_A, real_A)*10.0

            recovered_B = netG_A2B(fake_A)
            loss_cycle_BAB = criterion_cycle(recovered_B, real_B)*10.0

            # Total loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB
            loss_G.backward()

            optimizer_G.step()
            ###################################

            ###### Discriminator A ######
            optimizer_D_A.zero_grad()

            # Real loss
            pred_real = netD_A(real_A)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_A = fake_A_buffer.push_and_pop(fake_A)
            pred_fake = netD_A(fake_A.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_A = (loss_D_real + loss_D_fake)*0.5
            loss_D_A.backward()

            optimizer_D_A.step()
            ###################################

            ###### Discriminator B ######
            optimizer_D_B.zero_grad()

            # Real loss
            pred_real = netD_B(real_B)
            loss_D_real = criterion_GAN(pred_real, target_real)

            # Fake loss
            fake_B = fake_B_buffer.push_and_pop(fake_B)
            pred_fake = netD_B(fake_B.detach())
            loss_D_fake = criterion_GAN(pred_fake, target_fake)

            # Total loss
            loss_D_B = (loss_D_real + loss_D_fake)*0.5
            loss_D_B.backward()

            optimizer_D_B.step()
            ###################################

            loss_dict["loss_G"] += loss_G.item()
            loss_dict["loss_G_identity"] += (loss_identity_A.item() + loss_identity_B.item())
            loss_dict["loss_G_GAN"] += (loss_GAN_A2B.item() + loss_GAN_B2A.item())
            loss_dict["loss_G_cycle"] += (loss_cycle_ABA.item() + loss_cycle_BAB.item())
            loss_dict["loss_D"] += (loss_D_A.item() + loss_D_B.item())

            # Progress report (http://localhost:8097)
            # logger.log({'loss_G': loss_G, 'loss_G_identity': (loss_identity_A + loss_identity_B), 'loss_G_GAN': (loss_GAN_A2B + loss_GAN_B2A),
            #             'loss_G_cycle': (loss_cycle_ABA + loss_cycle_BAB), 'loss_D': (loss_D_A + loss_D_B)},
            #             images={'real_A': real_A, 'real_B': real_B, 'fake_A': fake_A, 'fake_B': fake_B})

        sum_loss = 0
        for key in loss_dict:
            loss_dict[key] /= len(train_dataset)
            sum_loss += loss_dict[key]

        print("epoch:", epoch, "loss:", loss_dict, "sum loss:", sum_loss)

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

        if epoch%10==0:
            overall_score = test(netG_A2B, netG_B2A, netD_A, netD_B, overall_score=overall_score)

if __name__ == '__main__':
    train()
