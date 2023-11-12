import torch
from chat_anything.face_generator.pipelines.lpw_stable_diffusion import StableDiffusionLongPromptWeightingPipeline 

@torch.no_grad()
def generate(pipe, prompt, negative_prompt, **generating_conf):
    pipe_longprompt = StableDiffusionLongPromptWeightingPipeline(
        unet=pipe.unet,
        text_encoder=pipe.text_encoder,
        vae=pipe.vae,
        tokenizer=pipe.tokenizer,
        scheduler=pipe.scheduler, 
        safety_checker=None,
        feature_extractor=None,
    )
    print('generating: ', prompt)
    print('using negative prompt: ', negative_prompt)
    embeds = pipe_longprompt._encode_prompt(prompt=prompt, negative_prompt=negative_prompt, device=pipe.device, num_images_per_prompt=1, do_classifier_free_guidance=generating_conf['guidance_scale']>1,)
    negative_prompt_embeds, prompt_embeds = embeds.split(embeds.shape[0]//2)
    pipe_out = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        **generating_conf,
    )
    return pipe_out

if __name__ == '__main__':
    from diffusers.pipelines import StableDiffusionPipeline
    import argparse
    def main():
        parser = argparse.ArgumentParser()
        parser.add_argument(
            '--prompts',type=str,default=['starry night','Impression Sunrise, drawn by Claude Monet'], nargs='*'
        )
        
        args = parser.parse_args()
        prompts = args.prompts
        print(f'generating {prompts}')
        model_id = 'pretrained_model/sd-v1-4'
        pipe = StableDiffusionPipeline.from_pretrained(model_id,).to('cuda')
        images = pipe(prompts).images
        for i, image in enumerate(images):
            image.save(f'{prompts[i]}_{i}.png')

    main()
    