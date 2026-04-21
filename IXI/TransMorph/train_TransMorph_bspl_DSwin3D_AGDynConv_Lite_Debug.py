from torch.utils.tensorboard import SummaryWriter
import glob
import json
import os
import sys
from collections import defaultdict

import losses
import matplotlib.pyplot as plt
import models.transformation as transformation
import numpy as np
import torch
import utils
from data import datasets, trans
from natsort import natsorted
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

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
        self.terminal.flush()
        self.log.flush()


class JsonMetricLogger(object):
    def __init__(self, path):
        self.handle = open(path, "a", encoding="utf-8")

    def write(self, record):
        self.handle.write(json.dumps(record, ensure_ascii=True) + "\n")
        self.handle.flush()

    def close(self):
        self.handle.close()


class DebugMetricAccumulator(object):
    def __init__(self):
        self.storage = defaultdict(list)

    def update(self, metrics):
        for key, value in metrics.items():
            self.storage[key].append(float(value))

    def mean(self):
        return {key: float(np.mean(values)) for key, values in self.storage.items()}


def build_block_stage_lookup(depths):
    lookup = []
    for stage_idx, depth in enumerate(depths):
        for block_idx in range(depth):
            lookup.append((stage_idx, block_idx))
    return lookup


def tensor_batch_mean(tensor):
    tensor = tensor.detach().float()
    if tensor.ndim == 0:
        return [float(tensor.cpu().item())]
    if tensor.ndim == 1:
        return [float(x) for x in tensor.cpu().tolist()]
    reduced = tensor.mean(dim=0)
    return [float(x) for x in reduced.reshape(-1).cpu().tolist()]


def add_vector_metrics(metrics, stage_store, block_prefix, stage_name, values):
    for index, value in enumerate(values):
        metrics[f"{block_prefix}/{stage_name}_{index}"] = float(value)
        stage_store[f"{stage_name}_{index}"].append(float(value))


def collect_debug_metrics(attn_aux_list, stage_depths):
    metrics = {}
    stage_lookup = build_block_stage_lookup(stage_depths)
    stage_store = defaultdict(lambda: defaultdict(list))

    for global_block_idx, block_aux in enumerate(attn_aux_list):
        stage_idx, block_idx = stage_lookup[global_block_idx]
        block_prefix = f"stage{stage_idx}/block{block_idx}"
        current_stage = stage_store[stage_idx]

        offset_mag = block_aux["offset_magnitude"].detach().float()
        offset_mag_mean = float(offset_mag.mean().cpu().item())
        offset_mag_max = float(offset_mag.max().cpu().item())
        metrics[f"{block_prefix}/offset_mag_mean"] = offset_mag_mean
        metrics[f"{block_prefix}/offset_mag_max"] = offset_mag_max
        current_stage["offset_mag_mean"].append(offset_mag_mean)
        current_stage["offset_mag_max"].append(offset_mag_max)

        if "offset_component_abs_mean" in block_aux:
            offset_xyz = tensor_batch_mean(block_aux["offset_component_abs_mean"])
        else:
            offset_xyz = tensor_batch_mean(block_aux["offsets"].abs().mean(dim=(1, 2, 3, 4)))
        axis_names = ("x", "y", "z")
        for axis_name, axis_value in zip(axis_names, offset_xyz):
            metric_key = f"{block_prefix}/offset_abs_{axis_name}"
            metrics[metric_key] = axis_value
            current_stage[f"offset_abs_{axis_name}"].append(axis_value)

        branch_energy = tensor_batch_mean(block_aux["branch_energy"])
        add_vector_metrics(metrics, current_stage, block_prefix, "branch_energy", branch_energy)

        attn_entropy = tensor_batch_mean(block_aux.get("attn_branch_entropy", torch.zeros(1, 1)))
        add_vector_metrics(metrics, current_stage, block_prefix, "attn_entropy", attn_entropy)

        attn_peak = tensor_batch_mean(block_aux.get("attn_branch_peak", torch.zeros(1, 1)))
        add_vector_metrics(metrics, current_stage, block_prefix, "attn_peak", attn_peak)

        gate_weights = tensor_batch_mean(block_aux.get("dynconv_gate_weights", torch.zeros(1, 1)))
        add_vector_metrics(metrics, current_stage, block_prefix, "dynconv_gate", gate_weights)
        current_stage["dynconv_gate_max"].append(max(gate_weights))

        branch_response = tensor_batch_mean(block_aux.get("dynconv_branch_response", torch.zeros(1, 1)))
        add_vector_metrics(metrics, current_stage, block_prefix, "dynconv_branch_response", branch_response)

        guide_vector = block_aux["guide_vector"].detach().float()
        guide_norm = float(torch.linalg.vector_norm(guide_vector, dim=1).mean().cpu().item())
        metrics[f"{block_prefix}/guide_norm"] = guide_norm
        current_stage["guide_norm"].append(guide_norm)

        num_offset_heads = offset_mag.shape[1]
        guide_offset_mean = float(guide_vector[:, :num_offset_heads].mean().cpu().item())
        guide_branch_mean = float(guide_vector[:, num_offset_heads:].mean().cpu().item())
        metrics[f"{block_prefix}/guide_offset_mean"] = guide_offset_mean
        metrics[f"{block_prefix}/guide_branch_mean"] = guide_branch_mean
        current_stage["guide_offset_mean"].append(guide_offset_mean)
        current_stage["guide_branch_mean"].append(guide_branch_mean)

        feat_norm = float(block_aux.get("dynconv_feat_norm", torch.zeros(1, 1)).detach().float().mean().cpu().item())
        guidance_norm = float(block_aux.get("dynconv_guidance_norm", torch.zeros(1, 1)).detach().float().mean().cpu().item())
        modulation_energy = float(block_aux.get("ffn_modulation_energy", torch.zeros(1, 1)).detach().float().mean().cpu().item())
        metrics[f"{block_prefix}/dynconv_feat_norm"] = feat_norm
        metrics[f"{block_prefix}/dynconv_guidance_norm"] = guidance_norm
        metrics[f"{block_prefix}/ffn_modulation_energy"] = modulation_energy
        current_stage["dynconv_feat_norm"].append(feat_norm)
        current_stage["dynconv_guidance_norm"].append(guidance_norm)
        current_stage["ffn_modulation_energy"].append(modulation_energy)

    all_offset_values = []
    for stage_idx, stage_metrics in stage_store.items():
        for metric_name, values in stage_metrics.items():
            stage_value = float(np.mean(values))
            metrics[f"stage{stage_idx}/{metric_name}"] = stage_value
            if metric_name == "offset_mag_mean":
                all_offset_values.append(stage_value)

    if all_offset_values:
        metrics["global/offset_mag_mean"] = float(np.mean(all_offset_values))
    return metrics


def write_metrics_to_tensorboard(writer, prefix, metrics, step):
    for key, value in metrics.items():
        writer.add_scalar(f"{prefix}/{key}", value, step)


def format_debug_snapshot(metrics, num_stages):
    parts = []
    for stage_idx in range(num_stages):
        offset_key = f"stage{stage_idx}/offset_mag_mean"
        gate_key = f"stage{stage_idx}/dynconv_gate_max"
        if offset_key not in metrics:
            continue
        offset_value = metrics[offset_key]
        gate_value = metrics.get(gate_key, 0.0)
        parts.append(f"s{stage_idx}:off={offset_value:.3e},gate={gate_value:.3f}")
    return " | ".join(parts)


def main():
    batch_size = 1
    atlas_dir = 'Path_to_IXI_data/atlas.pkl'
    train_dir = 'Path_to_IXI_data/Train/'
    val_dir = 'Path_to_IXI_data/Val/'
    weights = [1, 1]

    config = CONFIGS_TM['TransMorphBSpline-DSwin3D-AGDynConv-Lite']
    save_dir = 'TransMorphBSpline_DSwin3D_AGDynConv_Lite_Debug_ncc_{}_diffusion_{}_offmag_{}_offsmooth_{}/'.format(
        weights[0],
        weights[1],
        config.offset_mag_weight,
        config.offset_smooth_weight,
    )
    if not os.path.exists('experiments/' + save_dir):
        os.makedirs('experiments/' + save_dir)
    if not os.path.exists('logs/' + save_dir):
        os.makedirs('logs/' + save_dir)
    sys.stdout = Logger('logs/' + save_dir)
    debug_json_logger = JsonMetricLogger('logs/' + save_dir + 'debug_metrics.jsonl')

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
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=updated_lr, weight_decay=0, amsgrad=True)
    criterions = [losses.NCC_vxm(), losses.Grad3d(penalty='l2')]
    best_dsc = 0
    writer = SummaryWriter(log_dir='logs/' + save_dir)
    global_step = 0

    epoch_pbar = tqdm(range(epoch_start, max_epoch), desc='Total Epochs', position=0)
    for epoch in epoch_pbar:
        print(f'\n--- Epoch {epoch} Training Starts ---')
        loss_all = utils.AverageMeter()
        sim_all = utils.AverageMeter()
        flow_reg_all = utils.AverageMeter()
        off_mag_all = utils.AverageMeter()
        off_smooth_all = utils.AverageMeter()

        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch} [Train]', position=1, leave=False)
        for batch_idx, data in enumerate(train_pbar):
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

            global_step += 1

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
        print('Epoch {} Average Train Loss {:.4f}'.format(epoch, loss_all.avg))

        eval_dsc = utils.AverageMeter()
        val_debug_meter = DebugMetricAccumulator()

        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch} [Val]', position=1, leave=False)
        with torch.no_grad():
            for batch_idx, data in enumerate(val_pbar):
                model.eval()
                data = [t.cuda() for t in data]
                x = data[0]
                y = data[1]
                x_seg = data[2]
                y_seg = data[3]
                grid_img = mk_grid_img(8, 1, config.img_size)
                _, _, disp, aux = model((x, y))
                val_debug_meter.update(collect_debug_metrics(aux['attn_aux'], config.depths))
                with torch.cuda.device(GPU_iden):
                    def_out = transformation.warp(x_seg.cuda().float(), disp.cuda(), interp_mode='nearest')
                    def_grid = transformation.warp(grid_img.float(), disp.cuda(), interp_mode='bilinear')
                dsc = utils.dice_val_VOI(def_out.long(), y_seg.long())
                eval_dsc.update(dsc.item(), x.size(0))
                val_pbar.set_postfix({'DSC': f'{dsc.item():.4f}'})

        print('Epoch {} Average Val DSC {:.4f}'.format(epoch, eval_dsc.avg))

        val_debug_epoch = val_debug_meter.mean()
        write_metrics_to_tensorboard(writer, 'DebugValEpoch', val_debug_epoch, epoch)
        debug_json_logger.write({
            'mode': 'val_epoch',
            'epoch': epoch,
            'metrics': val_debug_epoch,
        })
        print('[Debug][Val][Epoch {} Summary] {}'.format(
            epoch,
            format_debug_snapshot(val_debug_epoch, len(config.depths)),
        ))

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
    debug_json_logger.close()


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


if __name__ == '__main__':
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
