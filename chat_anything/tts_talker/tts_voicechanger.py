import random
import shutil
import os
from gradio_client import Client
client = Client("http://127.0.0.1:7860/")
# How to use this new TTS Client? I leave the gradio api demo page as a reference
# client = Client("http://127.0.0.1:7860/")
# result = client.predict(
# 				"Howdy!",	# str in '请填写您想要转换的文本(中英皆可)' Textbox component
# 				"Bilibili - 一清清清,Bilibili - 一清清清",	# str (Option from: [('Bilibili - 一清清清', 'Bilibili - 一清清清'), ('ALL - Bob Sponge', 'ALL - Bob Sponge'), ('ALL - Ariana Grande', 'ALL - Ariana Grande'), ('ALL - Stefanie Sun', 'ALL - Stefanie Sun')])
# 								in '请选择您的AI歌手(必选)' Dropdown component
#  				"Microsoft Adri Online (Natural) - Afrikaans (South Africa) (Female),Microsoft Adri Online (Natural) - Afrikaans (South Africa) (Female)",	# str (Option from: [('Microsoft Adri Online (Natural) - Afrikaans (South Africa) (Female)', 'Microsoft Adri Online (Natural) - Afrikaans (South Africa) (Female)'), ('Microsoft Willem Online (Natural) - Afrikaans (South Africa) (Male)', 'Microsoft Willem Online (Natural) - Afrikaans (South Africa) (Male)'), ('Microsoft Anila Online (Natural) - Albanian (Albania) (Female)', 'Microsoft Anila Online (Natural) - Albanian (Albania) (Female)'), ('Microsoft Ilir Online (Natural) - Albanian (Albania) (Male)', 'Microsoft Ilir Online (Natural) - Albanian (Albania) (Male)'), ('Microsoft Ameha Online (Natural) - Amharic (Ethiopia) (Male)', 'Microsoft Ameha Online (Natural) - Amharic (Ethiopia) (Male)'), ('Microsoft Mekdes Online (Natural) - Amharic (Ethiopia) (Female)', 'Microsoft Mekdes Online (Natural) - Amharic (Ethiopia) (Female)'),
#  ('Microsoft Amina Online (Natural) - Arabic (Algeria) (Female)', 'Microsoft Amina Online (Natural) - Arabic (Algeria) (Female)'), ('Microsoft Ismael Online (Natural) - Arabic (Algeria) (Male)', 'Microsoft Ismael Online (Natural) - Arabic (Algeria) (Male)'), ('Microsoft Ali Online (Natural) - Arabic (Bahrain) (Male)', 'Microsoft Ali Online (Natural) - Arabic (Bahrain) (Male)'), ('Microsoft Laila Online (Natural) - Arabic (Bahrain) (Female)', 'Microsoft Laila Online (Natural) - Arabic (Bahrain) (Female)'), ('Microsoft Salma Online (Natural) - Arabic (Egypt) (Female)', 'Microsoft Salma Online (Natural) - Arabic (Egypt) (Female)'), ('Microsoft Shakir Online (Natural) - Arabic (Egypt) (Male)', 'Microsoft Shakir Online (Natural) - Arabic (Egypt) (Male)'), ...)
# 
# 								in '请选择一个相应语言的说话人' Dropdown component
# 				-24,	# int | float (numeric value between -24 and 24)
# 								in 'Pitch' Slider component
# 				"pm",	# str in 'f0 methods' Radio component
# 				0,	# int | float (numeric value between 0 and 1)
# 								in 'Feature ratio' Slider component
# 				0,	# int | float (numeric value between 0 and 7)
# 								in 'Filter radius' Slider component
# 				0,	# int | float (numeric value between 0 and 1)
# 								in 'Volume envelope mix rate' Slider component
# 				"Disable resampling,Disable resampling",	# str (Option from: [('Disable resampling', 'Disable resampling'), ('16000', '16000'), ('22050', '22050'), ('44100', '44100'), ('48000', '48000')])
# 								in 'Resample rate' Dropdown component
# 				api_name="/tts_conversion"
# )
# print(result)

TTS_MODELS = {
    "male":{
        "Chinese": "Microsoft Yunyang Online (Natural) - Chinese (Mainland) (Male)",
        "English": "Microsoft Eric Online (Natural) - English (United States) (Male)",
        "Japanese": "Microsoft Keita Online (Natural) - Japanese (Japan) (Male)",
    },
    "female":{
        "Chinese": "Microsoft Xiaoyi Online (Natural) - Chinese (Mainland) (Female)",
        "English": "Microsoft Ana Online (Natural) - English (United States) (Female)",
        "Japanese": "Microsoft Nanami Online (Natural) - Japanese (Japan) (Female)",
    }
}


class TTSTalker():
    def __init__(self,selected_voice, gender, language) -> None:
        self.selected_voice = selected_voice
        self.gender = gender
        self.language = language
        
    def test(self, text,  audio_path=None):
        self.gender = random.choice(['male', 'female']) if self.gender not in TTS_MODELS else self.gender
        languages = TTS_MODELS[self.gender].keys()
        self.language = random.choice(languages) if self.language not in languages else self.language
        tts_model = TTS_MODELS[self.gender][self.language]
        result = client.predict(
            text,	# str in '请填写您想要转换的文本(中英皆可)' Textbox component
            self.selected_voice,	# str (Option from: [('Bilibili - 一清清清', 'Bilibili - 一清清清'), ('ALL - Bob Sponge', 'ALL - Bob Sponge'), ('ALL - Ariana Grande', 'ALL - Ariana Grande'), ('ALL - Stefanie Sun', 'ALL - Stefanie Sun')]) in '请选择您的AI歌手(必选)' Dropdown component
            tts_model, # in '请选择一个相应语言的说话人' Dropdown component
            0,	# int | float (numeric value between -24 and 24) in 'Pitch' Slider component
            "pm",	# str in 'f0 methods' Radio component
            0,	# int | float (numeric value between 0 and 1) in 'Feature ratio' Slider component
            0,	# int | float (numeric value between 0 and 7) in 'Filter radius' Slider component
            0,	# int | float (numeric value between 0 and 1) in 'Volume envelope mix rate' Slider component
            "Disable resampling",	# str (Option from: [('Disable resampling', 'Disable resampling'), ('16000','16000'), ('22050', '22050'), ('44100', '44100'), ('48000', '48000')]) in 'Resample rate' Dropdown component
            api_name="/tts_conversion"
        )
        print(result[1])
        print(result)
        if result[1] == 'Success':
            if not os.path.exists(audio_path):
                os.makedirs(audio_path)
            output_path = os.path.join(audio_path, 'tempfile.mp3')
            print(output_path)
            shutil.copy(result[0], output_path)
            return output_path
        else:
            raise ValueError("failed with SVC")
