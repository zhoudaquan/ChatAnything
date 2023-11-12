# This class stores Azure voice data. Specifically, the class stores several records containing
# language, lang_code, gender, voice_id and engine. The class also has a method to return the
# voice_id, lang_code and engine given a language and gender.

NEURAL_ENGINE = "neural"
STANDARD_ENGINE = "standard"


class AzureVoiceData:
    def get_voice(self, language, gender):
        for voice in self.voice_data:
            if voice['language'] == language and voice['gender'] == gender:
                return voice['azure_voice']
        return None

    def __init__(self):
        self.voice_data = [
            {'language': 'Arabic',
             'azure_voice': 'ar-EG-ShakirNeural',
             'gender': 'Male'},
            {'language': 'Arabic (Gulf)',
             'azure_voice': 'ar-KW-FahedNeural',
             'gender': 'Male'},
            {'language': 'Catalan',
             'azure_voice': 'ca-ES-EnricNeural',
             'gender': 'Male'},
            {'language': 'Chinese (Cantonese)',
             'azure_voice': 'yue-CN-YunSongNeural',
             'gender': 'Male'},
            {'language': 'Chinese (Mandarin)',
             'azure_voice': 'zh-CN-YunxiNeural',
             'gender': 'Male'},
            {'language': 'Danish',
             'azure_voice': 'da-DK-JeppeNeural',
             'gender': 'Male'},
            {'language': 'Dutch',
             'azure_voice': 'nl-NL-MaartenNeural',
             'gender': 'Male'},
            {'language': 'English (Australian)',
             'azure_voice': 'en-AU-KenNeural',
             'gender': 'Male'},
            {'language': 'English (British)',
             'azure_voice': 'en-GB-RyanNeural',
             'gender': 'Male'},
            {'language': 'English (Indian)',
             'azure_voice': 'en-IN-PrabhatNeural',
             'gender': 'Male'},
            {'language': 'English (New Zealand)',
             'azure_voice': 'en-NZ-MitchellNeural',
             'gender': 'Male'},
            {'language': 'English (South African)',
             'azure_voice': 'en-ZA-LukeNeural',
             'gender': 'Male'},
            {'language': 'English (US)',
             'azure_voice': 'en-US-ChristopherNeural',
             'gender': 'Male'},
            {'language': 'English (Welsh)',
             'azure_voice': 'cy-GB-AledNeural',
             'gender': 'Male'},
            {'language': 'Finnish',
             'azure_voice': 'fi-FI-HarriNeural',
             'gender': 'Male'},
            {'language': 'French',
             'azure_voice': 'fr-FR-HenriNeural',
             'gender': 'Male'},
            {'language': 'French (Canadian)',
             'azure_voice': 'fr-CA-AntoineNeural',
             'gender': 'Male'},
            {'language': 'German',
             'azure_voice': 'de-DE-KlausNeural',
             'gender': 'Male'},
            {'language': 'German (Austrian)',
             'azure_voice': 'de-AT-JonasNeural',
             'gender': 'Male'},
            {'language': 'Hindi',
             'azure_voice': 'hi-IN-MadhurNeural',
             'gender': 'Male'},
            {'language': 'Icelandic',
             'azure_voice': 'is-IS-GunnarNeural',
             'gender': 'Male'},
            {'language': 'Italian',
             'azure_voice': 'it-IT-GianniNeural',
             'gender': 'Male'},
            {'language': 'Japanese',
             'azure_voice': 'ja-JP-KeitaNeural',
             'gender': 'Male'},
            {'language': 'Korean',
             'azure_voice': 'ko-KR-GookMinNeural',
             'gender': 'Male'},
            {'language': 'Norwegian',
             'azure_voice': 'nb-NO-FinnNeural',
             'gender': 'Male'},
            {'language': 'Polish',
             'azure_voice': 'pl-PL-MarekNeural',
             'gender': 'Male'},
            {'language': 'Portuguese (Brazilian)',
             'azure_voice': 'pt-BR-NicolauNeural',
             'gender': 'Male'},
            {'language': 'Portuguese (European)',
             'azure_voice': 'pt-PT-DuarteNeural',
             'gender': 'Male'},
            {'language': 'Romanian',
             'azure_voice': 'ro-RO-EmilNeural',
             'gender': 'Male'},
            {'language': 'Russian',
             'azure_voice': 'ru-RU-DmitryNeural',
             'gender': 'Male'},
            {'language': 'Spanish (European)',
             'azure_voice': 'es-ES-TeoNeural',
             'gender': 'Male'},
            {'language': 'Spanish (Mexican)',
             'azure_voice': 'es-MX-LibertoNeural',
             'gender': 'Male'},
            {'language': 'Spanish (US)',
             'azure_voice': 'es-US-AlonsoNeural"',
             'gender': 'Male'},
            {'language': 'Swedish',
             'azure_voice': 'sv-SE-MattiasNeural',
             'gender': 'Male'},
            {'language': 'Turkish',
             'azure_voice': 'tr-TR-AhmetNeural',
             'gender': 'Male'},
            {'language': 'Welsh',
             'azure_voice': 'cy-GB-AledNeural',
             'gender': 'Male'},
        ]


# Run from the command-line
if __name__ == '__main__':
    azure_voice_data = AzureVoiceData()

    azure_voice = azure_voice_data.get_voice('English (US)', 'Male')
    print('English (US)', 'Male', azure_voice)

    azure_voice = azure_voice_data.get_voice('English (US)', 'Female')
    print('English (US)', 'Female', azure_voice)

    azure_voice = azure_voice_data.get_voice('French', 'Female')
    print('French', 'Female', azure_voice)

    azure_voice = azure_voice_data.get_voice('French', 'Male')
    print('French', 'Male', azure_voice)

    azure_voice = azure_voice_data.get_voice('Japanese', 'Female')
    print('Japanese', 'Female', azure_voice)

    azure_voice = azure_voice_data.get_voice('Japanese', 'Male')
    print('Japanese', 'Male', azure_voice)

    azure_voice = azure_voice_data.get_voice('Hindi', 'Female')
    print('Hindi', 'Female', azure_voice)

    azure_voice = azure_voice_data.get_voice('Hindi', 'Male')
    print('Hindi', 'Male', azure_voice)
