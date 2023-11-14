
import os
from PIL import Image
import random
import shutil
import datetime
import torchvision.transforms.functional as f
import torch

from typing import Optional, Tuple
from threading import Lock
from langchain import ConversationChain

from chat_anything.tts_talker.tts_edge import TTSTalker
from chat_anything.sad_talker.sad_talker import SadTalker
from chat_anything.chatbot.chat import load_chain
from chat_anything.chatbot.select import model_selection_chain
from chat_anything.chatbot.voice_select import voice_selection_chain
import gradio as gr


TALKING_HEAD_WIDTH = "350"
sadtalker_checkpoint_path = "MODELS/SadTalker"
config_path = "chat_anything/sad_talker/config"

class ChatWrapper:
    def __init__(self):
        self.lock = Lock()
        self.sad_talker = SadTalker(
            sadtalker_checkpoint_path, config_path, lazy_load=True)

    def __call__(
            self,
            api_key: str,
            inp: str,
            history: Optional[Tuple[str, str]],
            chain: Optional[ConversationChain],
            speak_text: bool, talking_head: bool,
            uid: str,
            talker : None,
            fullbody : str,
    ):
        """Execute the chat functionality."""
        self.lock.acquire()
        if chain is None:
            history.append((inp, "Please register with your API key first!"))
        else:
            try:
                print("\n==== date/time: " + str(datetime.datetime.now()) + " ====")
                print("inp: " + inp)
                print("speak_text: ", speak_text)
                print("talking_head: ", talking_head)
                history = history or []
                # If chain is None, that is because no API key was provided.
                output = "Please paste your OpenAI key from openai.com to use this app. " + \
                    str(datetime.datetime.now())

                output = chain.predict(input=inp).strip()
                output = output.replace("\n", "\n\n")

                text_to_display = output

                # #预定义一个talker
                # talker = MaleEn()
                history.append((inp, text_to_display))

                html_video, temp_file, html_audio, temp_aud_file = None, None, None, None
                if speak_text:
                    if talking_head:
                        html_video, temp_file = self.do_html_video_speak(
                         talker, output, fullbody, uid)
                    else:
                        html_audio, temp_aud_file = self.do_html_audio_speak(
                         talker,  output,uid)
                else:
                    if talking_head:
                        temp_file = os.path.join('tmp', uid, 'videos')
                        html_video = create_html_video(
                            temp_file, TALKING_HEAD_WIDTH)
                    else:
                        pass

            except Exception as e:
                raise e
            finally:
                self.lock.release()
        return history, history, html_video, temp_file, html_audio, temp_aud_file, ""
    

    def do_html_audio_speak(self,talker, words_to_speak, uid):
        audio_path = os.path.join('tmp', uid, 'audios')
        print('uid:', uid, ":", words_to_speak)
        audo_file_path = talker.test(text=words_to_speak, audio_path=audio_path)
        html_audio = '<pre>no audio</pre>'
        try:
            temp_aud_file = gr.File(audo_file_path)
            # print("audio-----------------------------------------------------success")
            temp_aud_file_url = "/file=" + temp_aud_file.value['name']
            html_audio = f'<audio autoplay><source src={temp_aud_file_url} type="audio/mp3"></audio>'
        except IOError as error:
            # Could not write to file, exit gracefully
            print(error)
            return None, None

        return html_audio, audo_file_path

    def do_html_video_speak(self,talker,words_to_speak,fullbody, uid):
        if fullbody:
            # preprocess='somthing'
            preprocess='full'
        else:
            preprocess='crop'
        print("success")
        video_path = os.path.join('tmp', uid, 'videos')
        if not os.path.exists(video_path):
            os.makedirs(video_path)
        video_file_path = os.path.join(video_path, 'tempfile.mp4')
        _, audio_path = self.do_html_audio_speak(
            talker,words_to_speak,uid)
        face_file_path = os.path.join('tmp', uid, 'images', 'test.jpg')
        
        video = self.sad_talker.test(face_file_path, audio_path,preprocess, uid=uid) #video_file_path
        # print("---------------------------------------------------------success")
        # print(f"moving {video} -> {video_file_path}")
        shutil.move(video, video_file_path)

        return video_file_path, video_file_path


    def generate_init_face_video(self,class_concept="clock", llm=None,uid=None,fullbody=None, ref_image=None, seed=None):
        """
        """
        print('generate concept of', class_concept)
        print("=================================================")
        print('fullbody:', fullbody)
        print('uid:', uid)
        print("==================================================")
        chain, memory, personality_text = load_chain(llm, class_concept)
        model_conf, selected_model = model_selection_chain(llm, class_concept, conf_file='resources/models.yaml') # use class concept to choose a generating model, otherwise crack down
        # model_conf, selected_model = model_selection_chain(llm, personality_text, conf_file='resources/models_personality.yaml') # use class concept to choose a generating model, otherwise crack down
        voice_conf, selected_voice = model_selection_chain(llm, personality_text, conf_file='resources/voices_edge.yaml')

        # added for safe face generation
        print('generate concept of', class_concept)
        augment_word_list = ["Female ", "female ", "beautiful ", "small ", "cute "]
        first_sentence = "Hello, how are you doing today?"
        voice_conf, selected_voice = model_selection_chain(llm, personality_text, conf_file='resources/voices_edge.yaml')
        talker = TTSTalker(selected_voice=selected_voice, gender=voice_conf['gender'], language=voice_conf['language'])
        model_conf, selected_model = model_selection_chain(llm, class_concept, conf_file='resources/models.yaml') # use class concept to choose a generating model, otherwise crack down
        retry_cnt = 4
        if ref_image is None:
            face_files = os.listdir(FACE_DIR)
            face_img_path = os.path.join(FACE_DIR, random.choice(face_files))
            ref_image = Image.open(face_img_path)

        print('loading face generating model')
        anything_facemaker = load_face_generator(
            model_dir=model_conf['model_dir'],                                                                                           
            lora_path=model_conf['lora_path'],                                                                                           
            prompt_template=model_conf['prompt_template'],                                                                               
            negative_prompt=model_conf['negative_prompt'],    
        )
        retry_cnt = 0                                                                                                                                  
        has_face = anything_facemaker.has_face(ref_image)
        init_strength = 1.0 if has_face else 0.85                                                                                       
        strength_retry_step = -0.04 if has_face else 0.04
        while retry_cnt < 8:                                                                                                
            try:                                                                                                                                 
                generate_face_image(                                                                                                             
                    anything_facemaker,
                    class_concept,
                    ref_image,
                    uid=uid,                                                                                                  
                    strength=init_strength if (retry_cnt==0 and has_face) else init_strength + retry_cnt * strength_retry_step,                                          
                    controlnet_conditioning_scale=0.5 if retry_cnt == 8 else 0.3,
                    seed=seed,                                                                                                                              
                )                                                                                                                                
                self.do_html_video_speak(talker, first_sentence, fullbody, uid=uid)                                                                   
                video_file_path = os.path.join('tmp', uid, 'videos/tempfile.mp4')                                                                
                htm_video = create_html_video(                                                                                                   
                    video_file_path, TALKING_HEAD_WIDTH)                                                                                                                                                                                                     
                break                                                                                                                            
            except Exception as e:                                                                                                               
                retry_cnt += 1                                                                                                                
                class_concept = random.choice(augment_word_list) + class_concept                                                                                                                                                                            
                print(e)         
        # end of repeat block       

        return chain, memory, htm_video, talker


    def update_talking_head(self, widget, uid, state):
        # print("success----------------")
        if widget:
            state = widget
            temp_file = os.path.join('tmp', uid, 'videos')
            video_html_talking_head = create_html_video(
                temp_file, TALKING_HEAD_WIDTH)
            return state, video_html_talking_head
        else:
            return None, "<pre></pre>"


def reset_memory(history, memory):
    memory.clear()
    history = []
    return history, history, memory
            

def create_html_video(file_name, width):
    return file_name


def create_html_audio(file_name):
    if os.path.exists(file_name):
        tmp_audio_file = gr.File(file_name, visible=False)
        tmp_aud_file_url = "/file=" + tmp_audio_file.value['name']
        html_audio = f'<audio><source src={tmp_aud_file_url} type="audio/mp3"></audio>'
        del tmp_aud_file_url
    else:
       html_audio = f'' 
    
    return html_audio


def update_foo(widget, state):
    if widget:
        state = widget
        return state


# Pertains to question answering functionality
def update_use_embeddings(widget, state):
    if widget:
        state = widget
        return state

# This is the code for image generating.


def load_face_generator(model_dir, lora_path, prompt_template, negative_prompt):
    from chat_anything.face_generator.long_prompt_control_generator import LongPromptControlGenerator
    # # using local
    model_zoo = "MODELS"
    face_control_dir = os.path.join(
        model_zoo, "Face-Landmark-ControlNet", "models_for_diffusers")
    face_detect_path = os.path.join(
        model_zoo, "SadTalker", "shape_predictor_68_face_landmarks.dat")
    # use remote, hugginface auto-download.
    # use your model path, has to be a model derived from stable diffusion v1-5
    anything_facemaker = LongPromptControlGenerator(
        model_dir=model_dir,
        lora_path=lora_path,
        prompt_template=prompt_template,
        negative_prompt=negative_prompt,
        face_control_dir=face_control_dir,
        face_detect_path=face_detect_path,
    )
    anything_facemaker.load_model(safety_checker=None)
    return anything_facemaker



FACE_DIR="resources/images/faces"
def generate_face_image(
        anything_facemaker,
        class_concept, 
        face_img_pil,
        uid=None,
        controlnet_conditioning_scale=1.0,
        strength=0.95,
        seed=42,
    ):
    face_img_pil = f.center_crop(
        f.resize(face_img_pil, 512), 512).convert('RGB')
    prompt = anything_facemaker.prompt_template.format(class_concept)
    # # There are four ways to generate a image by now.
    # pure_generate = anything_facemaker.generate(prompt=prompt, image=face_img_pil, do_inversion=False)
    # inversion = anything_facemaker.generate(prompt=prompt, image=face_img_pil, strength=strength, do_inversion=True)

    print('USING SEED:', seed)
    generator = torch.Generator(device=anything_facemaker.face_control_pipe.device)
    generator.manual_seed(seed)
    if strength is None:
        pure_control = anything_facemaker.face_control_generate(prompt=prompt, face_img_pil=face_img_pil, do_inversion=False,
                                                                 controlnet_conditioning_scale=controlnet_conditioning_scale, generator=generator)
        init_face_pil = pure_control
    else:
        control_inversion = anything_facemaker.face_control_generate(prompt=prompt, face_img_pil=face_img_pil, do_inversion=True, 
                                                                 strength=strength,
                                                                 controlnet_conditioning_scale=controlnet_conditioning_scale, generator=generator)
        init_face_pil = control_inversion
    print('succeeded generating face image')
    face_path = os.path.join('tmp', uid, 'images')
    if not os.path.exists(face_path):
        os.makedirs(face_path)
    # TODO: reproduce the images for return, shouldn't use the filesystem
    face_file_path = os.path.join(face_path, 'test.jpg')
    init_face_pil.save(face_file_path)
    return init_face_pil
