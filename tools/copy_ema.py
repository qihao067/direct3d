import argparse
import os
import torch
from collections import OrderedDict

def parse_args():
    parser = argparse.ArgumentParser(description='Copy ema to weights')
    parser.add_argument('checkpoint', help='path of checkpoint containing ema')
    return parser.parse_args()

def copy_ema(sd):
    new_sd = OrderedDict()
    for k, v in sd.items():
        if '_ema' in k:
            orig_k = k.replace('_ema', '')
            new_sd[orig_k] = v
        elif k not in new_sd:
            new_sd[k] = v
    return new_sd

def main():
    args = parse_args()
    sd = torch.load(args.checkpoint, map_location='cpu')
    isCkpt = 'state_dict' in sd
    if isCkpt:
        ckpt = sd
        sd = sd['state_dict']
    new_sd = copy_ema(sd)

    save_path = args.checkpoint.replace('.pth', '_copyema.pth')
    if isCkpt:
        ckpt['state_dict'] = new_sd
        torch.save(ckpt, save_path)
    else:
        torch.save(new_sd, save_path)
    print(f"Save new ckpt to {save_path}")

if __name__ == '__main__':
    main()