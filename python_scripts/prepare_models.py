import os
import os.path as osp
import requests
import shutil
from huggingface_hub import snapshot_download, HfApi
# from diffusers.pipelines.stable_diffusion.convert_from_ckpt import download_from_original_stable_diffusion_ckpt
from facexlib.utils import load_file_from_url
from facexlib.detection import init_detection_model

def hf_download_dir(repo_id, dirname):
    api = HfApi()
    space_list = api.list_repo_files(repo_id=repo_id)
    target_list = [target for target in space_list if target.startswith(dirname) ]

    print(target_list)
    for filename in target_list:
        print(f'downloading {filename}')
        api.hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir='.',
            local_dir_use_symlinks=True,
        )
    

MODEL_DIR='./MODELS'
os.makedirs(MODEL_DIR, exist_ok=True)

def prepare_sadtalker_models():
    snapshot_download(repo_id='vinthony/SadTalker', local_dir=osp.join(MODEL_DIR, 'SadTalker'), local_dir_use_symlinks=True)
    load_file_from_url(
        url='https://github.com/xinntao/facexlib/releases/download/v0.1.0/alignment_WFLW_4HG.pth', 
        model_dir='facexlib/weights', 
        progress=True, file_name=None, save_dir=osp.join(MODEL_DIR, 'gfpgan/weights',))
    init_detection_model('retinaface_resnet50', half=False,device='cpu', model_rootpath=osp.join(MODEL_DIR, 'gfpgan/weights',))

def prepare_face_generator_models():
    # from all source repo
    # snapshot_download(repo_id="georgefen/Face-Landmark-ControlNet", local_dir=osp.join(MODEL_DIR, "Face-Landmark-ControlNet"), allow_patterns=["models_for_diffusers/*"], local_dir_use_symlinks=True)
    # snapshot_download(repo_id="runwayml/stable-diffusion-v1-5", local_dir=osp.join(MODEL_DIR, "stable-diffusion-v1-5"), allow_patterns=["*.bin", '*.json', '*.txt'], ignore_patterns=['safety_checker'],local_dir_use_symlinks=True)
    # snapshot_download(repo_id="xiaolxl/GuoFeng3", local_dir=osp.join(MODEL_DIR, "GuoFeng3"), allow_patterns=["*.bin", '*.json', '*.txt'], ignore_patterns=['safety_checker*'],local_dir_use_symlinks=True)
    # snapshot_download(repo_id="simhuangxi/MoXin", local_dir=osp.join(MODEL_DIR, "MoXin"),local_dir_use_symlinks=True)
    # snapshot_download(repo_id="diffusers/controlnet-canny-sdxl-1.0", local_dir=osp.join(MODEL_DIR, "controlnet-canny-sdxl-1.0"), ignore_patterns=['*.bin'], local_dir_use_symlinks=True)
    # snapshot_download(repo_id="stablediffusionapi/anything-v5", local_dir=osp.join(MODEL_DIR, "anything-v5"), ignore_patterns=['*.bin'], local_dir_use_symlinks=True)
    # snapshot_download(
    #         repo_id="ermu2001/ChatAnything", 
    #         local_dir='.', 
    #         local_dir_use_symlinks=True,
    #     )
    hf_download_dir('ermu2001/ChatAnything', 'MODELS')
        
if __name__ == "__main__":
    prepare_sadtalker_models()
    prepare_face_generator_models()