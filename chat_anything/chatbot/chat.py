import datetime
from chat_anything.chatbot.personality import generate_personality_prompt
from langchain.prompts import PromptTemplate
from langchain import ConversationChain
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
import os
import random
import string


def load_chain(llm, class_concept=None):
    chain = None
    memory = None
    personality_text = None
    print(llm)
    if llm:
        print("class_concept", class_concept)
        if class_concept is None:
            class_concept = 'AI assistant'
        person_template, personality_text = generate_personality_prompt(llm, class_concept)

        PROMPT_TEMPLATE = PromptTemplate(
            input_variables=["history", "input"],
            template=person_template,
        )

        chain = ConversationChain(
            prompt=PROMPT_TEMPLATE,
            llm=llm,
            verbose=False,
            memory=ConversationBufferMemory(ai_prefix="You"),
        )
        print("New concept done for ", class_concept)

    return chain, memory, personality_text



def set_openai_api_key(api_key, use_gpt4, history=None, max_tokens=1024):
    """Set the api key and return chain.
    If no api_key, then None is returned.
    """
    if api_key and api_key.startswith("sk-") and len(api_key) > 50:
        os.environ["OPENAI_API_KEY"] = api_key
        print("\n\n ++++++++++++++ Setting OpenAI API key ++++++++++++++ \n\n")
        print(str(datetime.datetime.now()) + ": Before OpenAI, OPENAI_API_KEY length: " + str(
            len(os.environ["OPENAI_API_KEY"])))

        if use_gpt4:
            llm = ChatOpenAI(
                temperature=0, max_tokens=max_tokens, model_name="gpt-4")
            print("Trying to use llm ChatOpenAI with gpt-4")
        else:
            print("Trying to use llm ChatOpenAI with gpt-3.5-turbo")
            llm = ChatOpenAI(temperature=0, max_tokens=max_tokens,
                             model_name="gpt-3.5-turbo")

        print(str(datetime.datetime.now()) + ": After OpenAI, OPENAI_API_KEY length: " + str(
            len(os.environ["OPENAI_API_KEY"])))

        print(str(datetime.datetime.now()) + ": After load_chain, OPENAI_API_KEY length: " + str(
            len(os.environ["OPENAI_API_KEY"])))
        os.environ["OPENAI_API_KEY"] = ""
        history = history or []
        history.append(['', '[SYSTEM] OPENAI_API_KEY has been set, you can generate your object and talk to it now!'])
        uid = ''.join(random.sample(string.ascii_lowercase + string.ascii_uppercase, 5))
        video_file_path = os.path.join('tmp', uid, 'videos/tempfile.mp4')
        audio_file_path = os.path.join('tmp', uid, 'audio/tempfile.mp3')
        return llm, use_gpt4, history, uid, video_file_path, audio_file_path
    return None, None, None, None, None, None