#!/bin/bash

tmux new-session -d -s fastchat_controller 'python -m fastchat.serve.controller'

tmux new-window -t fastchat_controller -n model_worker_0 'CUDA_VISIBLE_DEVICES=0 python3 -m fastchat.serve.model_worker --model-path HuggingFaceH4/zephyr-7b-beta --model-names "gpt-3.5-turbo,text-davinci-003,text-embedding-ada-002" '

tmux new-window -t fastchat_controller -n api_server 'python -m fastchat.serve.openai_api_server --host localhost --port 8000'

tmux attach-session -t fastchat_controller
