safetensors_path=$1
dump_path=$2
python3 python_scripts/convert_original_stable_diffusion_to_diffusers.py \
    --checkpoint_path $safetensors_path  \
    --dump_path $dump_path \
    --from_safetensors
