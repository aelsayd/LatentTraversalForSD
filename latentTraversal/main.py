import diffusers
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
import os 
import tensorflow as tf0
from PIL import Image
import random
import threading
diffusers.logging.set_verbosity_error()

negative_prompt = "Disfigured, ugly oversaturat, extra fingers, blur, oversaturated, deformed hands, mutant hands, fusioned hands, deformed, mutant, bad anatomy, out of focus, poorly drawn, childish, mangled"

def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols

    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid

def generate(prompt, num, out, width=512, height=512, rows=2, columns=2):
    device = "cuda"
    #model_id = "/run/media/von/Data/diffusion/seek.art_MEGA"
    model_id = "/home/von/stableDiff/textual_inversion_cat"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        revision="fp16",
        torch_dtype=torch.float16,
    ).to(device)
    pipe.enable_attention_slicing()


    generator = torch.Generator(device=device)
    
    latents = None
    seeds = []
    num_images = rows*columns
    for _ in range(num_images):
        # Get a new random seed, store it and use it as the generator state
        seed = generator.seed()
        seeds.append(seed)
        generator = generator.manual_seed(seed)
        
        image_latents = torch.randn(
            (1, pipe.unet.in_channels, height // 8, width // 8),
            generator = generator,
            device = device,
            dtype=torch.half
        )
        latents = image_latents if latents is None else torch.cat((latents, image_latents))

    images = pipe(
        [prompt] * num_images,
        guidance_scale=7.5,
        latents = latents,
        width = width,
        height = height,
        num_inference_steps=num,
        negative_prompt=[negative_prompt]*num_images
    ).images

    image_grid(images, rows, columns).save(out)
    print(seeds)
    evolveForSequence(seeds[0], prompt, "t")
    return seeds

def evolveForSequence(seed, prompt, out, width=512, height=512, device="cuda", rows=1, cols=1):
    print(seed)
    device = "cuda"
    model_id = "/run/media/von/Data/diffusion/seek.art_MEGA"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        revision="fp16",
        torch_dtype=torch.float16,
    ).to(device)
    pipe.enable_attention_slicing()

    generator = torch.Generator(device=device)
 
    latents = None
    seeds = []
    num_images = rows*cols
    for _ in range(num_images):
        # Get a new random seed, store it and use it as the generator state
        generator = generator.manual_seed(seed)
        
        image_latents = torch.randn(
            (1, pipe.unet.in_channels, height // 8, width // 8),
            generator = generator,
            device = device,
            dtype=torch.half
        )
        for latent in image_latents:
            latent += random.uniform(0, 0.01)
        latents = image_latents if latents is None else torch.cat((latents, image_latents))

    for f in range(100):
        images = pipe(
            [prompt] * num_images,
            guidance_scale=7.5,
            latents = latents,
            width = width,
            height = height,
            num_inference_steps=20,
            negative_prompt=[negative_prompt]*num_images
        ).images

        image_grid(images, rows, cols).save(out+str(f)+".png")
        latents = EvolveLatents(latents)

def EvolveLatents(image_latents):
    latents = None
    for latent in image_latents:
        latent += random.uniform(-0.01, 0.01)
    latents = image_latents if latents is None else torch.cat((latents, image_latents))
    return latents
