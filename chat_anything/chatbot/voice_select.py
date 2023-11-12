from langchain import LLMChain
from langchain.prompts import PromptTemplate
from omegaconf import OmegaConf
import datetime

VOICE_SELECTION_PROMPT_TEMPLATE = """
Select one of the following voice based on the given concept.
You must choose one voice name based on the description of each model and the concept.


Cencept: {concept}

Voice name and description: {model_list}

Warning: {warning}

The avilable voice names: 
{model_name_list}

Selected voice name:
"""

GENDER_SELECTION_PROMPT_TEMPLATE = """
Select one of the following gender based on the given concept.
You must choose one gender based on the description of the concept. You must choose one gender Even if you can't decide.

Gender:
male
female

Cencept: {concept}
Selected gender male or female:
"""

LANGUAGE_SELECTION_PROMPT_TEMPLATE = """
Select one of the following language based on the given concept.
You must choose the language that is used by the description of the concept.

Languages:
Chinese
English
Japanese

Cencept: {concept}
Selected language:
"""

def load_voice_model_list():
    models_config = OmegaConf.load('resources/voices.yaml')
    models_dict = models_config['models']
    print(models_dict)
    model_list_str = ''
    model_name_list_str = ''
    for key, value in models_dict.items():
        model_list_str+="model name: " +key+', model description: '+value['desc']+'\n'
        model_name_list_str += key + ' '
    model_name_list_str += '\n'
    return model_list_str, models_dict, model_name_list_str

def get_vioce_model_chain(llm, class_concept):
    model_template = PromptTemplate(
        input_variables=["model_list", "concept", "model_name_list", "warning"],
        template=VOICE_SELECTION_PROMPT_TEMPLATE,
    )
    model_list_str, models_dict, model_name_list_str = load_voice_model_list()

    personality_chain = LLMChain(
        llm=llm, prompt=model_template, verbose=True)

    selected_model = None
    while (selected_model is None) or not (selected_model in models_dict):
        if (selected_model is not None) and not (selected_model in models_dict):
            warning_str = '{} is not in Model list! \n'.format(selected_model)
        else:
            warning_str = ''
        selected_model = personality_chain.run({'concept': class_concept, 'model_list':model_list_str, 'warning': warning_str, 'model_name_list': model_name_list_str})
    print("Selected model name: ", selected_model)
    
    return selected_model

def get_gender_chain(llm, class_concept):
    model_template = PromptTemplate(
        input_variables=["concept"],
        template=GENDER_SELECTION_PROMPT_TEMPLATE,
    )

    personality_chain = LLMChain(
        llm=llm, prompt=model_template, verbose=True)
    selected_gender = personality_chain.run({'concept': class_concept})
    print("Selected gender: ", selected_gender)
    return selected_gender

def get_language_chain(llm, class_concept):
    model_template = PromptTemplate(
        input_variables=["concept"],
        template=LANGUAGE_SELECTION_PROMPT_TEMPLATE,
    )

    personality_chain = LLMChain(
        llm=llm, prompt=model_template, verbose=True)
    selected_language = personality_chain.run({'concept': class_concept})
    print("Selected language: ", selected_language)
    return selected_language
                


def voice_selection_chain(llm, class_concept=None):
    chain = None
    memory = None
    if llm:
        print("class_concept", class_concept)
        if class_concept is None:
            class_concept = 'AI assistant'
        selected_model = get_vioce_model_chain(llm, class_concept)
        gender = get_gender_chain(llm, class_concept)
        language = get_language_chain(llm, class_concept)

    return selected_model, gender, language

