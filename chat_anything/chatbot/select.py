from langchain import LLMChain
from typing import OrderedDict
from langchain.prompts import PromptTemplate
from omegaconf import OmegaConf
import datetime

SELECTION_TEMPLATE = """
{concept}

Model name and description:
{option_list}

Warning: {warning}

The avilable Options: 
{choices}
Answer:
"""


def selection_chain(llm, class_concept, prompt, options):
    chain = None
    memory = None
    if llm:
        print("class_concept", class_concept)
        if class_concept is None:
            class_concept = 'AI assistant'
        prompt_template = prompt + SELECTION_TEMPLATE
        template = PromptTemplate(
            input_variables=["concept", "option_list", "warning", "choices"],
            template=prompt_template,
        )
        chain = LLMChain(
            llm=llm, prompt=template, verbose=True)
        print(options)
        option_list = [
            f"{chr(ord('A') + i)}. {conf['desc']}" for i, conf in enumerate(options.values())
        ]
        option_list = '\n'.join(option_list)
        selected_model = None

        warning_str = 'Choose from the available Options.'
        choices = ' '.join(chr(ord('A') + i) for i in range(len(options)))
        choice = chain.run({'concept': class_concept, 'option_list':option_list, 'warning': warning_str, 'choices': choices})
        print(f"LLM Responds (First character was used as the choice):{choice}", )
        choice = choice[0]

        selected_model = list(options.keys())[ord(choice) - ord('A')]
        print("Selected model name: ", selected_model)

    return selected_model

def model_selection_chain(llm, class_concept=None, conf_file='resources/models_personality.yaml'):
    chain = None
    memory = None
    if llm:
        print("class_concept", class_concept)
        if class_concept is None:
            class_concept = 'AI assistant'
        selection_config = OmegaConf.load(conf_file)
        selected_model = selection_chain(llm, class_concept, selection_config['prompt'], selection_config['models'])
        model_conf = selection_config['models'][selected_model]
    return model_conf, selected_model
