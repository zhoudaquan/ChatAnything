import PIL
from PIL import Image
from PIL import ImageDraw
import numpy as np

import dlib
import cv2
import torch

import diffusers
from diffusers import StableDiffusionPipeline, DiffusionPipeline
from diffusers import ControlNetModel, StableDiffusionControlNetPipeline, StableDiffusionControlNetImg2ImgPipeline
from chat_anything.face_generator.pipelines.lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline, get_weighted_text_embeddings 
from diffusers.schedulers import EulerAncestralDiscreteScheduler,DPMSolverMultistepScheduler # DPM++ SDE Karras

from chat_anything.face_generator.utils.generate import generate

from .long_prompt_generator import LongPromptGenerator

def draw_landmarks(image, landmarks, color="white", radius=2.5):
    draw = ImageDraw.Draw(image)
    for dot in landmarks:
        x, y = dot
        draw.ellipse((x-radius, y-radius, x+radius, y+radius), fill=color)

def get_ldmk_img(w, h, ldmks) -> PIL.Image:
        con_img = Image.new('RGB', (w, h), color=(0, 0, 0))
        draw_landmarks(con_img, ldmks)
        return con_img

class LongPromptControlGenerator(LongPromptGenerator):

    def __init__(self, model_dir, lora_path, prompt_template, negative_prompt, face_control_dir, face_detect_path,):
        self.face_control_dir = face_control_dir
        self.face_detect_path = face_detect_path
        super().__init__(model_dir, lora_path, prompt_template, negative_prompt)

    def load_model(self, *args, **kwargs):
        super().load_model(*args, **kwargs)
        self.face_detector = dlib.get_frontal_face_detector()
        self.face_predictor = dlib.shape_predictor(self.face_detect_path)
        # load control net
        face_controlnet = ControlNetModel.from_pretrained(self.face_control_dir).to('cuda', dtype=torch.float16)
        self.face_control_pipe = StableDiffusionControlNetPipeline(controlnet=face_controlnet, **self.pipe.components)
        self.face_control_img2img_pipe = StableDiffusionControlNetImg2ImgPipeline(controlnet=face_controlnet, **self.pipe.components)

    def _get_68landmarks_seq(self, img_np):
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        faces = self.face_detector(gray)
        landmarks = []
        for face in faces:
            shape = self.face_predictor(gray, face)
            for i in range(68):
                x = shape.part(i).x
                y = shape.part(i).y
                landmarks.append((x, y))
        return landmarks
   
    def has_face(self, img_pil):
        img_np = np.array(img_pil)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        faces = self.face_detector(gray)
        return len(faces) != 0
    
    def face_control_generate(
            self,
            prompt,
            face_img_pil,
            do_inversion=False,
            **kwargs,
        ):
        """
        Face control generating.
        """
        face_img_np = np.array(face_img_pil)
        ldmk_seq = self._get_68landmarks_seq(face_img_np)
        ldmk_img_pil = get_ldmk_img(face_img_pil.size[0], face_img_pil.size[1], ldmk_seq)      
        print('GENERATING:', prompt)

        generating_conf = {
            "prompt": prompt,
            "negative_prompt": self.negative_prompt,
            "num_inference_steps": 25,
            "guidance_scale": 7,
            "controlnet_conditioning_scale": kwargs.pop('controlnet_conditioning_scale', 1.0),
            "generator": kwargs.pop('generator', None),
        }
        
        if not do_inversion:
            generating_conf.update({
                "pipe": self.face_control_pipe,
                "image": ldmk_img_pil,
                "controlnet_conditioning_scale": kwargs.pop('controlnet_conditioning_scale', 1.0),
            })
        else:
            generating_conf.update({
                "pipe": self.face_control_img2img_pipe,
                "image": face_img_pil,
                "control_image": ldmk_img_pil,
                "strength": kwargs.pop('strength', 0.9),
            })
        pipe_out = generate(**generating_conf)
        generated_img = pipe_out[0][0]
        return generated_img