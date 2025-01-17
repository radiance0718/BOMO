from functools import partial
import os
import argparse
import yaml

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from guided_diffusion.condition_methods import get_conditioning_method
from guided_diffusion.measurements import get_noise, get_operator
from guided_diffusion.unet import create_model
from guided_diffusion.gaussian_diffusion import create_sampler
from data.dataloader import get_dataset, get_dataloader
from util.img_utils import clear_color, clear_gray, mask_generator
from util.logger import get_logger
from guided_diffusion.Model import getUnet


def crop_region(img, position, size):
    """根据位置裁剪图像区域"""
    if position == 'top_left':
        return img[:, :, :size, :size]
    elif position == 'top_right':
        return img[:, :, :size, -size:]
    elif position == 'bottom_left':
        return img[:, :, -size:, :size]
    elif position == 'bottom_right':
        return img[:, :, -size:, -size:]
    else:
        raise ValueError("Invalid position argument")

def load_yaml(file_path: str) -> dict:
    with open(file_path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_config', type=str)
    parser.add_argument('--diffusion_config', type=str)
    parser.add_argument('--task_config', type=str)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--save_dir', type=str, default='./results')
    args = parser.parse_args()
   
    # logger
    logger = get_logger()
    
    # Device setting
    device_str = f"cuda:{args.gpu}" if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device set to {device_str}.")
    device = torch.device(device_str)  
    
    # Load configurations
    model_config = load_yaml(args.model_config)
    diffusion_config = load_yaml(args.diffusion_config)
    task_config = load_yaml(args.task_config)
    
    # Load model
    model = getUnet(**model_config)
    model = model.to(device)
    model.load_state_dict(torch.load(model_config['model_path'], map_location=device))
    model.eval()

    # Prepare Operator and noise
    measure_config = task_config['measurement']
    operator = get_operator(device=device, **measure_config['operator'])
    noiser = get_noise(**measure_config['noise'])
    logger.info(f"Operation: {measure_config['operator']['name']} / Noise: {measure_config['noise']['name']}")

    # Prepare conditioning method
    cond_config = task_config['conditioning']
    cond_method = get_conditioning_method(cond_config['method'], operator, noiser, **cond_config['params'])
    measurement_cond_fn = cond_method.conditioning
    logger.info(f"Conditioning method : {task_config['conditioning']['method']}")
   
    # Load diffusion sampler
    sampler = create_sampler(**diffusion_config) 
    sample_fn = partial(sampler.p_sample_loop, model=model, measurement_cond_fn=measurement_cond_fn)
   
    # Working directory
    out_path = os.path.join(args.save_dir, measure_config['operator']['name'])
    os.makedirs(out_path, exist_ok=True)
    for img_dir in ['input', 'recon', 'progress', 'label']:
        os.makedirs(os.path.join(out_path, img_dir), exist_ok=True)

    # Prepare dataloader
    data_config = task_config['data']
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (0.5,))])
    dataset = get_dataset(**data_config, transforms=transform)
    loader = get_dataloader(dataset, batch_size=1, num_workers=0, train=False)

    # Exception) In case of inpainting, we need to generate a mask 
    if measure_config['operator']['name'] == 'inpainting':
        mask_gen = mask_generator(
           **measure_config['mask_opt']
        )
        

    for i, ref_img in enumerate(loader):
        logger.info(f"Inference for image {i}")
        fname = str(i).zfill(5) + '.png'
        ref_img = ref_img.to(device)
        y_n = ref_img
        # Sampling
        x_start = torch.randn(ref_img.shape, device=device).requires_grad_()
        print(x_start.shape, y_n.shape)
        sample = sample_fn(x_start=x_start, measurement=y_n, record=True, save_root=out_path)
        print("sample complete")
        print(sample.shape, y_n.shape, ref_img.shape)
        plt.imsave(os.path.join(out_path, 'input', fname), clear_gray(y_n), cmap = 'gray')
        plt.imsave(os.path.join(out_path, 'label', fname), clear_gray(ref_img), cmap = 'gray')
        plt.imsave(os.path.join(out_path, 'recon', fname), clear_gray(sample), cmap = 'gray')

if __name__ == '__main__':
    main()
