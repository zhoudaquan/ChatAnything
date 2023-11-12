ckpt_path=$1
dump_path=$2
original_config_path=" "
python python_scripts/convert_original_controlnet_to_diffusers.py \
    --checkpoint_path $ckpt_path  \
    --original_config_file "None" \
    --dump_path $dump_path \
    --device cpu