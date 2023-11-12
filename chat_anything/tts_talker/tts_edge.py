import random
import shutil
import os

import asyncio
import random

import edge_tts
from edge_tts import VoicesManager
import uuid
import shutil
# # How to use this api
# #!/usr/bin/env python3
# """
# Example of dynamic voice selection using VoicesManager.
# """
# import asyncio
# import random
# import edge_tts
# from edge_tts import VoicesManager
# TEXT = "Hoy es un buen día."
# OUTPUT_FILE = "spanish.mp3"
# async def amain() -> None:
#     """Main function"""
#     voices = await VoicesManager.create()
#     voice = voices.find(Gender="Male", Language="es")
#     # Also supports Locales
#     # voice = voices.find(Gender="Female", Locale="es-AR")
#     communicate = edge_tts.Communicate(TEXT, random.choice(voice)["Name"])
#     await communicate.save(OUTPUT_FILE)
# if __name__ == "__main__":
#     loop = asyncio.get_event_loop_policy().get_event_loop()
#     try:
#         loop.run_until_complete(amain())
#     finally:
#         loop.close()




class TTSTalker():
    def __init__(self, selected_voice, gender, language) -> None:
        self.selected_voice = selected_voice
        self.gender = gender
        self.language = language
        self.voice = asyncio.run(self.get_voice(gender, language))
        
    async def get_voice(self, gender, language):
        voices = await VoicesManager.create()
        voices = voices.find(Gender=gender, Language=language)
        voice = random.choice(voices)["Name"]
        return voice

    async def amain(self, text, file, voice) -> None:
        """Main function"""

        # Also supports Locales
        # voice = voices.find(Gender="Female", Locale="es-AR")
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(file)

        
    def test(self, text,  audio_path=None):
        if not os.path.exists(audio_path):
            os.mkdir(audio_path)
        voice_uuid = str(uuid.uuid4())[:5] + '.wav'
        audio_file = os.path.join(audio_path, voice_uuid)
        asyncio.run(self.amain(text, audio_file, self.voice))
        return audio_file

        
if __name__ == "__main__":
    audio_dir = 'test'
    tts_talker = TTSTalker('', 'Male', 'en').test('hello', audio_dir)
    tts_talker = TTSTalker('', 'Male', 'zh').test('hello', audio_dir)
    tts_talker = TTSTalker('', 'Female', 'en').test('hello',audio_dir)
    tts_talker = TTSTalker('', 'Female', 'zh').test('hello', audio_dir)