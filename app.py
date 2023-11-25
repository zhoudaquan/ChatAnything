import os
import ssl
import sys

import gradio as gr

import warnings
import whisper
from chat_anything.polly_utils import PollyVoiceData
from chat_anything.azure_utils import AzureVoiceData
from chat_anything.chatbot.chat import set_openai_api_key
from utils import ChatWrapper, update_foo, reset_memory

from python_scripts.prepare_models import prepare_face_generator_models, prepare_sadtalker_models
prepare_sadtalker_models()
prepare_face_generator_models()


ssl._create_default_https_context = ssl._create_unverified_context


TALKING_HEAD_WIDTH = "350"

LOOPING_TALKING_HEAD = "resources/videos/tempfile.mp4"

USE_GPT4_DEFAULT = False
FULLBODY_DEFAULT = False
POLLY_VOICE_DATA = PollyVoiceData()
AZURE_VOICE_DATA = AzureVoiceData()

# Pertains to WHISPER functionality
WHISPER_DETECT_LANG = "Detect language"

INSTRUCTION_MARKDOWN = """
# ChatAnything: Facetime Chat with LLM-Enhanced Personas
### DEMO INSTRUCTION
##### 0. Register
Input a OpenAI API Key of your own. This would be used to chat with openai-chatgpt. Make sure to disable the key afterwards🥹.
##### 1. Generate The init face😀 along with first chat
Input a Concept in the "Talking object" text box, then click on Generate button. The init face generation and module selection will be performed and used for the rest of this chat. Wait for a while and the video would be produced and played. Write simple concept for generating. The concept will be place on each prompt template for deciding the main concepts.
##### 2. Keep on Chatting🤑
Go on speak with the character. The init face and module selection will not reperform itself, now you are only chatting with the LM, along with the rendering of sadtalker. Hopefully, the API will not impose an excessive charge for this.


### FEATURES
##### 1. Upload a image for control/inversion starting point. Try some none face images and see how it works!
##### 2. seeding is provided. However if not providing a input image, there would be a random chosen facial landmark image for generating, which might include some randomness.
##### 3. Try out the examples.
##### 4. Say something and recorded your voice for a real facetime chat. Whisper will handle your voice, see setting-Whisper STT options.
##### 5. Decide whether to use the crop face out option, this will crop out the face from the generated image and render. This is promising for better animation rendering, however sometimes the croped image loses some elementary features of you intended concept.

"""

# UNCOMMENT TO USE WHISPER
warnings.filterwarnings("ignore")
WHISPER_MODEL = whisper.load_model("tiny")
print("WHISPER_MODEL", WHISPER_MODEL)


# UNCOMMENT TO USE WHISPER
def transcribe(aud_inp, whisper_lang):
    if aud_inp is None:
        return ""
    aud = whisper.load_audio(aud_inp)
    aud = whisper.pad_or_trim(aud)
    mel = whisper.log_mel_spectrogram(aud).to(WHISPER_MODEL.device)
    _, probs = WHISPER_MODEL.detect_language(mel)
    options = whisper.DecodingOptions()
    if whisper_lang != WHISPER_DETECT_LANG:
        whisper_lang_code = POLLY_VOICE_DATA.get_whisper_lang_code(
            whisper_lang)
        options = whisper.DecodingOptions(language=whisper_lang_code)
    result = whisper.decode(WHISPER_MODEL, mel, options)
    print("result.text", result.text)
    result_text = ""
    if result and result.text:
        result_text = result.text
    return result_text


chat = ChatWrapper()


with gr.Blocks() as block:
    llm_state = gr.State()
    history_state = gr.State()
    chain_state = gr.State()
    talker_state = gr.State()
    fullbody_state = gr.State(True)
    speak_text_state = gr.State(True)
    talking_head_state = gr.State(True)
    uid_state = gr.State()
    video_file_path = gr.State()
    audio_file_path = gr.State()

    memory_state = gr.State()


    # Pertains to WHISPER functionality
    whisper_lang_state = gr.State(WHISPER_DETECT_LANG)
    use_gpt4_state = gr.State(USE_GPT4_DEFAULT)

    with gr.Column():
        with gr.Row():
            gr.Markdown(INSTRUCTION_MARKDOWN)
        with gr.Row():  
            openai_api_key_textbox = gr.Textbox(placeholder="Paste your OpenAI API key (sk-...) or Keep Empty to use Local LLM",
                                            show_label=True, lines=1, type='password', value='', label='OpenAI API key')
            openai_api_key_register = gr.Button(
                value="Register").style(full_width=False)
            uid_textbox = gr.Textbox(show_label=True, value=uid_state, lines=1, label='UID')
            seed = gr.Slider(
                label="Seed",
                minimum=-1,
                maximum=2147483647,
                step=1,
                randomize=True,
            )

    with gr.Tab("Chat"):
        with gr.Row():        
            with gr.Column(scale=1, min_width=TALKING_HEAD_WIDTH, visible=True):
                with gr.Column():
                    class_prompt = gr.Textbox(
                        'apple',
                        default='apple',
                        type="text", label='Talking object'
                    )
                    init_face_btn = gr.Button(
                        value="Generate").style(full_width=False)

                my_file = gr.File(label="Upload a file",
                                  type="file", visible=False)

                # video_html = gr.HTML('')
                video_html = gr.Video(label="Generated Video", autoplay=True)

                ref_image = gr.Image(
                    type="pil",
                    interactive=True,
                    label="Image: Upload your image.",
                )
                tmp_aud_file = gr.File(
                    type="file", visible=False)
                audio_html = gr.HTML('')
                init_face_btn.click(chat.generate_init_face_video, inputs=[class_prompt, llm_state, uid_state,fullbody_state, ref_image, seed],
                                    outputs=[chain_state, memory_state, video_html,talker_state])


            with gr.Column(scale=7):
                chatbot = gr.Chatbot()


                message = gr.Textbox(label="What's on your mind??",
                                     placeholder="What's the answer to life, the universe, and everything?",
                                     lines=1)
                submit = gr.Button(value="Send", variant="secondary").style(
                    full_width=False)

                audio_comp = gr.Microphone(source="microphone", type="filepath", label="Just say it!",
                                           interactive=True, streaming=False)
                audio_comp.change(transcribe, inputs=[
                                  audio_comp, whisper_lang_state], outputs=[message])


        with gr.Accordion("General examples", open=False):
            gr.Examples(
                examples=[
                    ["cyberpunk godess", "Who are you?", "resources/images/annie.jpg", 393212389],
                    ["unbelievable beauty fairy", "Who are you?", "resources/images/lenna.jpg", 222679277],
                    ["tree monster", "Who are you?", None],
                    ["pineapple monster", "Who are you?", None],
                    ["tricky Polaris", "Who are you?", None, 1670155100],
                    ["watermelon", "Who are you?", "resources/images/watermelon.jpg", 42],
                ],
                inputs=[class_prompt, message, ref_image, seed],
            )

    with gr.Tab("Settings"):
        with gr.Tab("General"):

            talking_head_cb = gr.Checkbox(
                label="Show talking head", value=True)
            talking_head_cb.change(chat.update_talking_head, inputs=[talking_head_cb, uid_state, talking_head_state],
                                   outputs=[talking_head_state, video_html])

            use_gpt4_cb = gr.Checkbox(label="Use GPT-4 (experimental) if your OpenAI API has access to it",
                                      value=USE_GPT4_DEFAULT)

            fullbody_state = gr.Checkbox(label="Use full body instead of a face.",
                                      value=True)

            use_gpt4_cb.change(set_openai_api_key,
                               inputs=[openai_api_key_textbox,
                                       use_gpt4_cb],
                               outputs=[llm_state, use_gpt4_state, chatbot, uid_state, video_file_path, audio_file_path])

            reset_btn = gr.Button(value="Reset chat",
                                  variant="secondary").style(full_width=False)
            reset_btn.click(reset_memory, inputs=[history_state, memory_state],
                            outputs=[chatbot, history_state, memory_state])

            
        with gr.Tab("Whisper STT"):
            whisper_lang_radio = gr.Radio(label="Whisper speech-to-text language:", choices=[
                WHISPER_DETECT_LANG, "Arabic", "Arabic (Gulf)", "Catalan", "Chinese (Cantonese)", "Chinese (Mandarin)",
                "Danish", "Dutch", "English (Australian)", "English (British)", "English (Indian)", "English (New Zealand)",
                "English (South African)", "English (US)", "English (Welsh)", "Finnish", "French", "French (Canadian)",
                "German", "German (Austrian)", "Georgian", "Hindi", "Icelandic", "Indonesian", "Italian", "Japanese",
                "Korean", "Norwegian", "Polish",
                "Portuguese (Brazilian)", "Portuguese (European)", "Romanian", "Russian", "Spanish (European)",
                "Spanish (Mexican)", "Spanish (US)", "Swedish", "Turkish", "Ukrainian", "Welsh"],
                value=WHISPER_DETECT_LANG)

            whisper_lang_radio.change(update_foo,
                                      inputs=[whisper_lang_radio,
                                              whisper_lang_state],
                                      outputs=[whisper_lang_state])

    gr.HTML("""
        <p>This application is based on <a href='https://huggingface.co/spaces/JavaFXpert/Chat-GPT-LangChain/'>Chat-GPT-LangChain</a>, <a href='https://github.com/hwchase17/langchain'>LangChain</a>
        </p>""")

    message.submit(chat, inputs=[openai_api_key_textbox, message, history_state, chain_state,
                                 speak_text_state, talking_head_state, uid_state,talker_state,fullbody_state],
                    outputs=[chatbot, history_state, video_html, my_file, audio_html, tmp_aud_file, message])

    submit.click(chat, inputs=[openai_api_key_textbox, message, history_state, chain_state,
                               speak_text_state, talking_head_state, uid_state,talker_state,fullbody_state],
                outputs=[chatbot, history_state, video_html, my_file, audio_html, tmp_aud_file, message])

    openai_api_key_register.click(set_openai_api_key,
                                  inputs=[openai_api_key_textbox, 
                                          use_gpt4_state, chatbot],
                                  outputs=[llm_state, use_gpt4_state, chatbot, uid_state, video_file_path, audio_file_path])

if __name__ == "__main__":
    # import sys
    # if len(sys.argv) == 1:
    #     port = 8901
    # else:
    #     port = int(sys.argv[1])
    # block.launch(debug=True, server_name="0.0.0.0",
    #              server_port=port, share=True, enable_queue = True)
    block.launch(share=True)