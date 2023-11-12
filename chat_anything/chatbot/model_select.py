from langchain import LLMChain
from langchain.prompts import PromptTemplate
from omegaconf import OmegaConf
import datetime

MODEL_SELECTION_PROMPT_TEMPLATE = """
Select one of the following models based on the given concept.
You must choose one model name based on the description of each model and the concept!

Cencept: {concept}

Model name and description: {model_list}

Warning: {warning}

The avilable model names: 
{model_name_list}

Selected model name:
"""

def load_model_list():
    models_config = OmegaConf.load('resources/models.yaml')
    models_dict = models_config['models']
    model_name_list_str = ''
    print(models_dict)
    model_list_str = ''
    for key, value in models_dict.items():
        model_list_str+="model name: " +key+', model description: '+value['desc']+'\n'
        model_name_list_str += key + ' '
    model_name_list_str += '\n'
    return model_list_str, models_dict, model_name_list_str

def model_selection_chain(llm, class_concept=None):
    chain = None
    memory = None
    if llm:
        print("class_concept", class_concept)
        if class_concept is None:
            class_concept = 'AI assistant'


        template = PromptTemplate(
            input_variables=["model_list", "concept", "warning", "model_name_list"],
            template=MODEL_SELECTION_PROMPT_TEMPLATE,
        )
        model_list_str, models_dict, model_name_list_str = load_model_list()

        personality_chain = LLMChain(
            llm=llm, prompt=template, verbose=True)
        selected_model = None
        while (selected_model is None) or not (selected_model in models_dict):
            if (selected_model is not None) and not (selected_model in models_dict):
                warning_str = '{} is not in Model list! \n'.format(selected_model)
            else:
                warning_str = ''
            selected_model = personality_chain.run({'concept': class_concept, 'model_list':model_list_str, 'warning': warning_str, 'model_name_list': model_name_list_str})
        print("Selected model name: ", selected_model)

    return models_dict[selected_model]
