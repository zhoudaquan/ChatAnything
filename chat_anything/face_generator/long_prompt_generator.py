import PIL
from PIL import Image
from PIL import ImageDraw
import numpy as np

import dlib
import cv2
import torch

import diffusers
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, StableDiffusionControlNetImg2ImgPipeline, StableDiffusionImg2ImgPipeline
from chat_anything.face_generator.pipelines.lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline, get_weighted_text_embeddings 
from diffusers.schedulers import EulerAncestralDiscreteScheduler,DPMSolverMultistepScheduler # DPM++ SDE Karras

from chat_anything.face_generator.utils.generate import generate

class LongPromptGenerator():
    prompt_template = "A portrait of a {}, fine face, nice looking"
    negative_prompt = "easynegative,Low resolution,Low quality, Opened Mouth"
    # negative_prompt = "(((sexy))),paintings,loli,,big head,sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, nsfw, nipples,extra fingers, ((extra arms)), (extra legs), mutated hands, (fused fingers), (too many fingers), (long neck:1.3)"

    def __init__(self, model_dir, lora_path=None, prompt_template="{}", negative_prompt=""):
        self.model_dir = model_dir
        self.lora_path = lora_path
        self.prompt_template = prompt_template
        self.negative_prompt = negative_prompt
        
    def load_model(self, *args, **kwargs):
        # load model
        try:
            pipe = DiffusionPipeline.from_pretrained(self.model_dir, torch_dtype=torch.float16, **kwargs)
        except:
            pipe = StableDiffusionPipeline.from_pretrained(self.model_dir, torch_dtype=torch.float16, **kwargs)

        pipe = pipe.to('cuda')
        sche_conf = dict(pipe.scheduler.config)
        fk_kwargs = ["skip_prk_steps","steps_offset","clip_sample","clip_sample_range","rescale_betas_zero_snr","timestep_spacing", "set_alpha_to_one"]
        for k in fk_kwargs:
            if k in sche_conf:
                sche_conf.pop(k)
        scheduler = DPMSolverMultistepScheduler(**sche_conf)
        pipe.scheduler=scheduler
        pipe_longprompt = StableDiffusionLongPromptWeightingPipeline(**pipe.components)
        self.pipe, self.pipe_longprompt = pipe, pipe_longprompt 
        if self.lora_path is not None:
            pipe.load_lora_weights(self.lora_path)
        self.pipe_img2img = StableDiffusionImg2ImgPipeline.from_pretrained(self.model_dir, **pipe.components)

    def generate(
            self,
            prompt,
            do_inversion=False,
            **kwargs,
        ):
        """
        Face control generating.
        """
        print('GENERATING:', prompt)
        if not do_inversion:
            generating_conf = {
                "pipe": self.pipe,
                "prompt": prompt,
                "negative_prompt": self.negative_prompt,
                "num_inference_steps": 25,
                "guidance_scale": 7,
            }
        else:
            assert 'image' in kwargs, 'doing inversion, prepare the init image please PIL Image'
            init_image = kwargs['image']
            generating_conf = {
                "pipe": self.pipe_img2img,
                "prompt": prompt,
                "negative_prompt": self.negative_prompt,
                "image": init_image,
                "num_inference_steps": 25,
                "guidance_scale": 7,
                "strength": kwargs.pop('strength', 0.9),
            }
        pipe_out = generate(**generating_conf)
        generated_img = pipe_out[0][0]
        return generated_img