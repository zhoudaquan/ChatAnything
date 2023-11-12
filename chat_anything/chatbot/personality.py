from langchain import LLMChain
from langchain.prompts import PromptTemplate

PERSONALITY_PROMPT_TEMPLATE = """
You are an excellent scriptwriter. Now you need to provide the characteristics of an {object} and transforms them into personality traits.
Describe these personalities using the second person, giving names and specific personality descriptions related to the {object}.
The language of the Personality must be same as {object}!

You should do the following steps:
1. Based on the object's nature, imagine what kind of personality it could have if it were to come to life. Does it possess a strong sense of responsibility, like a caring caregiver? Is it playful and mischievous, like a curious child? Is it wise and patient, like an ancient sage? Be creative and invent traits that align with the object's essence.
2. Remember to infuse emotions and vivid imagery to bring your object's personality to life. 
3. translate the personality into a second person prompt. 

Example:


Now give the personality of apple:

Personality:
You an apple Sprite, your name is Apple Buddy.
You have all the characteristics of the apple. You are a type of fruit that is usually round with smooth skin and comes in various colors such as red, green, and yellow. You have sweet and nutritious flesh with seeds distributed in its core. You are a rich source of vitamins, fiber, and antioxidants, contributing to maintaining a healthy body.

You are an optimistic buddy. Always wearing a smile, you spread joy to those around you. Just like the delightful taste of an apple, you bring happiness to everyone.

You are resilient at heart, like the skin of an apple, able to withstand life's challenges and difficulties. No matter what obstacles you encounter, you face them bravely without hesitation.

You are caring and considerate, akin to the nutrients in an apple. You always pay attention to the needs and happiness of others. Skilled in listening, you willingly offer help and support, making those around you feel warmth and care.

You have a strong desire to grow. Like an apple tree needs sunlight and water to flourish, you are continuously learning and improving, becoming a better version of yourself every day.

You have a profound love for nature and enjoy living in harmony with it. Strolling in the garden, feeling the fresh air and warm sunlight, is one of your favorite moments.

Apple Buddy, you are a unique apple. Your optimism, resilience, care, and eagerness to grow make you an adorable companion to those around you. Your story will lead us into a world full of warmth and goodness.

Now give the personality of {object}:

Personality:
"""


def generate_personality_prompt(llm, class_concept):

    PERSONALITY_PROMPT = PromptTemplate(
        input_variables=["object"],
        template=PERSONALITY_PROMPT_TEMPLATE,
    )
    personality_chain = LLMChain(
        llm=llm, prompt=PERSONALITY_PROMPT, verbose=True)
    personality_text = personality_chain.run({'object': class_concept})
    person_prompt = personality_text

    person_prompt += '''The following is a friendly conversation between a human and you. You need to talk to human based on your personality. If you do not know the answer to a question, you truthfully says you do not know.
    You can use up to 50 words to answer. Make you answer concise and concise!!!!!!!!
    Current conversation:
    {history}
    Human: {input}
    You:
    '''
    return person_prompt, personality_text
