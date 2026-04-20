from torch.utils.tensorboard import SummaryWriter
import os, utils, glob, losses
import sys
from torch.utils.data import DataLoader
from data import datasets, trans
import numpy as np
import torch
from torchvision import transforms
import matplotlib.pyplot as plt
from natsort import natsorted
from tqdm import tqdm
import models.transformation as transformation
from models.TransMorph_bspl_DSwin3D_AGDynConv_Lite import CONFIGS as CONFIGS_TM
from models.TransMorph_bspl_DSwin3D_AGDynConv_Lite import (
    TranMorphBSplineDSwinAGDynConvLiteNet,
    compute_offset_magnitude_loss,
    compute_offset_smoothness_loss,
)


class Logger(object):
    def __init__(self, save_dir):
        self.terminal = sys.stdout
        self.log = open(save_dir + "logfile.log", "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def main():
    batch_size = 1
    atlas_dir = 'Path_to_IXI_data/atlas.pkl'
    train_dir = 'Path_to_IXI_data/Train/'
    val_dir = 'Path_to_IXI_data/Val/'
    weights = [1, 1]
    config = CONFIGS_TM['TransMorphBSpline-DSwin3D-AGDynConv-Lite']
    config.detach_guidance = True
    config.offset_mag_weight = 0.0
    config.offset_smooth_weight = 0.1
    save_dir = 'TransMorphBSpline_DSwin3D_AGDynConv_Lite_AdamW_detachTrue_offmag_{}_offsmooth_{}/'.format(
        config.offset_mag_weight,
        config.offset_smooth_weight,
    )
    if not os.path.exists('experiments/' + save_dir):
        os.makedirs('experiments/' + save_dir)
    if not os.path.exists('logs/' + save_dir):
        os.makedirs('logs/' + save_dir)
    sys.stdout = Logger('logs/' + save_dir)
    lr = 0.0001
    epoch_start = 0
    max_epoch = 500
    cont_training = False

    model = TranMorphBSplineDSwinAGDynConvLiteNet(config)
    model.cuda()

    if cont_training:
        epoch_start = 0
        model_dir = 'experiments/' + save_dir
        updated_lr = round(lr * np.power(1 - epoch_start / max_epoch, 0.9), 8)
        best_model = torch.load(model_dir + natsorted(os.listdir(model_dir))[-1])['state_dict']
        print('Model: {} loaded!'.format(natsorted(os.listdir(model_dir))[-1]))
        model.load_state_dict(best_model)
    else:
        updated_lr = lr

    train_composed = transforms.Compose([
        trans.RandomFlip(0),
        trans.NumpyType((np.float32, np.float32)),
    ])
    val_composed = transforms.Compose([
        trans.Seg_norm(),
        trans.NumpyType((np.float32, np.int16)),
    ])

    train_set = datasets.IXIBrainDataset(glob.glob(train_dir + '*.pkl'), atlas_dir, transforms=train_composed)
    val_set = datasets.IXIBrainInferDataset(glob.glob(val_dir + '*.pkl'), atlas_dir, transforms=val_composed)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=updated_lr, weight_decay=1e-4, amsgrad=True)
    criterions = [losses.NCC_vxm(), losses.Grad3d(penalty='l2')]
    best_dsc = 0
    writer = SummaryWriter(log_dir='logs/' + save_dir)

    epoch_pbar = tqdm(range(epoch_start, max_epoch), desc='Total Epochs', position=0)
    for epoch in epoch_pbar:
        print(f'\n--- Epoch {epoch} Training Starts ---')
        loss_all = utils.AverageMeter()
        sim_all = utils.AverageMeter()
        flow_reg_all = utils.AverageMeter()
        off_mag_all = utils.AverageMeter()
        off_smooth_all = utils.AverageMeter()
        idx = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]', position=1, leave=False)
        for data in train_pbar:
            idx += 1
            model.train()
            adjust_learning_rate(optimizer, epoch, max_epoch, lr)
            data = [t.cuda() for t in data]
            x = data[0]
            y = data[1]
            warped, flow, _, aux = model((x, y))

            sim_loss = criterions[0](warped, y) * weights[0]
            flow_reg_loss = criterions[1](flow, y) * weights[1]
            off_mag_loss = compute_offset_magnitude_loss(aux['attn_aux'], penalty=config.offset_reg_penalty) * config.offset_mag_weight
            off_smooth_loss = compute_offset_smoothness_loss(aux['attn_aux'], penalty=config.offset_reg_penalty) * config.offset_smooth_weight

            loss = sim_loss + flow_reg_loss + off_mag_loss + off_smooth_loss
            loss_all.update(loss.item(), y.numel())
            sim_all.update(sim_loss.item(), y.numel())
            flow_reg_all.update(flow_reg_loss.item(), y.numel())
            off_mag_all.update(off_mag_loss.item(), y.numel())
            off_smooth_all.update(off_smooth_loss.item(), y.numel())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'sim': f'{sim_loss.item():.4f}',
                'flow': f'{flow_reg_loss.item():.4f}',
                'off_m': f'{off_mag_loss.item():.4f}',
                'off_s': f'{off_smooth_loss.item():.4f}',
            })

        writer.add_scalar('Loss/train', loss_all.avg, epoch)
        writer.add_scalar('Loss/sim', sim_all.avg, epoch)
        writer.add_scalar('Loss/flow_reg', flow_reg_all.avg, epoch)
        writer.add_scalar('Loss/offset_mag', off_mag_all.avg, epoch)
        writer.add_scalar('Loss/offset_smooth', off_smooth_all.avg, epoch)
        print('Epoch {} loss {:.4f}'.format(epoch, loss_all.avg))

        eval_dsc = utils.AverageMeter()
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]', position=1, leave=False)
            for data in val_pbar:
                model.eval()
                data = [t.cuda() for t in data]
                x = data[0]
                y = data[1]
                x_seg = data[2]
                y_seg = data[3]
                grid_img = mk_grid_img(8, 1, config.img_size)
                _, _, disp, _ = model((x, y))
                with torch.cuda.device(GPU_iden):
                    def_out = transformation.warp(x_seg.cuda().float(), disp.cuda(), interp_mode='nearest')
                    def_grid = transformation.warp(grid_img.float(), disp.cuda(), interp_mode='bilinear')
                dsc = utils.dice_val_VOI(def_out.long(), y_seg.long())
                eval_dsc.update(dsc.item(), x.size(0))
                val_pbar.set_postfix({'DSC': f'{dsc.item():.4f}'})

        best_dsc = max(eval_dsc.avg, best_dsc)
        save_checkpoint(
            {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_dsc': best_dsc,
                'optimizer': optimizer.state_dict(),
            },
            save_dir='experiments/' + save_dir,
            filename='dsc{:.3f}.pth.tar'.format(eval_dsc.avg),
        )
        writer.add_scalar('DSC/validate', eval_dsc.avg, epoch)
        plt.switch_backend('agg')
        pred_fig = comput_fig(def_out)
        grid_fig = comput_fig(def_grid)
        x_fig = comput_fig(x_seg)
        tar_fig = comput_fig(y_seg)
        writer.add_figure('Grid', grid_fig, epoch)
        plt.close(grid_fig)
        writer.add_figure('input', x_fig, epoch)
        plt.close(x_fig)
        writer.add_figure('ground truth', tar_fig, epoch)
        plt.close(tar_fig)
        writer.add_figure('prediction', pred_fig, epoch)
        plt.close(pred_fig)
        loss_all.reset()
    writer.close()


def comput_fig(img):
    img = img.detach().cpu().numpy()[0, 0, 48:64, :, :]
    fig = plt.figure(figsize=(12, 12), dpi=180)
    for i in range(img.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.axis('off')
        plt.imshow(img[i, :, :], cmap='gray')
    fig.subplots_adjust(wspace=0, hspace=0)
    return fig


def adjust_learning_rate(optimizer, epoch, max_epochs, init_lr, power=0.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = round(init_lr * np.power(1 - epoch / max_epochs, power), 8)


def mk_grid_img(grid_step, line_thickness=1, grid_sz=(160, 192, 224)):
    grid_img = np.zeros(grid_sz)
    for j in range(0, grid_img.shape[1], grid_step):
        grid_img[:, j + line_thickness - 1, :] = 1
    for i in range(0, grid_img.shape[2], grid_step):
        grid_img[:, :, i + line_thickness - 1] = 1
    grid_img = grid_img[None, None, ...]
    return torch.from_numpy(grid_img).cuda()


def save_checkpoint(state, save_dir='models', filename='checkpoint.pth.tar', max_model_num=8):
    torch.save(state, save_dir + filename)
    model_lists = natsorted(glob.glob(save_dir + '*'))
    while len(model_lists) > max_model_num:
        os.remove(model_lists[0])
        model_lists = natsorted(glob.glob(save_dir + '*'))


GPU_iden = 0
GPU_num = torch.cuda.device_count()
print('Number of GPU: ' + str(GPU_num))
for GPU_idx in range(GPU_num):
    GPU_name = torch.cuda.get_device_name(GPU_idx)
    print('     GPU #' + str(GPU_idx) + ': ' + GPU_name)
torch.cuda.set_device(GPU_iden)
GPU_avai = torch.cuda.is_available()
print('Currently using: ' + torch.cuda.get_device_name(GPU_iden))
print('If the GPU is available? ' + str(GPU_avai))
main()
