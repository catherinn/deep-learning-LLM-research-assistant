import gradio as gr
import os
from models import MyVectorStore
from loguru import logger
from typing import List, Tuple

from utils import check_url
from collect_url import get_urls, extact_txt_from_urls
from logical_state import LogicProcess
from pathlib import Path

if Path('open_ai_key.txt').exists():
    os.environ['OPENAI_API_KEY'] = open(Path('open_ai_key.txt'), 'r').read()
    logger.info('Open AI key loaded')
else:
    logger.warning('No open_ai_key.txt to load the key.')

with gr.Blocks() as demo:

    meta = gr.State({'logic': LogicProcess()})

    chatbot = gr.Chatbot(value=[[None, 'Please provide a valid URL of a website that you wish to inquire about. \n'
                               'Attention: Extracting information from websites without their explicit permission can be '
                               'against the terms of use of many websites. Before undertaking any such action, make sure '
                               'to adhere to the policies of the respective website.\n'
                                'You can try with: https://www.gov.uk/government/organisations/department-for-work-pensions']])

    msg = gr.Textbox(placeholder="Ask me a question", show_label=False)

    with gr.Column(scale=1):
        clear_history = gr.Button("Delete conversation")

    def respond(message_user: str,  chat_history: List[Tuple[str, str]], meta: gr.State):
        logic = meta['logic']
        if logic.step == 0:
            is_working, error, logic = check_url(message_user, logic=logic)
            answer = f'{is_working} : {error}'
            if error != '200': # display only mistake
                chat_history[-1][1] = answer
            chat_history[-1][0] = message_user

        if logic.step == 1:
            # if website works: try to load the informations and add information messages to the user
            urls, _  = get_urls(logic.url, level=2, filtre_in=[logic.url],
                     filtre_not_in=[])
            if len(urls) == 0 : urls = [logic.url]

            chat_history.append([None, f'{len(urls)} web sites loaded. '])
            extact_txt_from_urls(url_library=urls, dir_output='to_del')
            logic.model_vector = MyVectorStore(dir_save='model_delet', dir_load='to_del')
            chat_history.append([None, f'Loading the vectors done'])
            chat_history.append([None, f"Let's ask some questions ?"])
            logic.step = 2
            logger.info('Ready to work.')
            return "", chat_history, meta

        if logic.step == 2:
            logger.info(f'Question: {message_user}')
            k = 2
            logger.info(f'k = {k}')
            scores, paragraph = logic.model_vector.get_closet_texts_from_vectorstore(message_user, k=k)
            logger.info(f'Accuracy Paragraph: {scores}')
            logger.info(f'Paragraph: {paragraph}')
            answer_llm = logic.model_vector.llm.answer_from_question(question=message_user, context=paragraph)
            chat_history.append((message_user, f"{answer_llm}"))

            return "", chat_history, meta

        return "", chat_history, meta

    def clear_data(meta: gr.State) -> Tuple[List[Tuple[str|None, str|None]], gr.State]:

        chat_history = [[None, 'Please provide a valid URL of a website that you wish to inquire about. \n'
                               'Attention: Extracting information from websites without their explicit permission can be '
                               'against the terms of use of many websites. Before undertaking any such action, make sure '
                               'to adhere to the policies of the respective website.\n'
                               'You can try with: https://www.gov.uk/government/organisations/department-for-work-pensions']]
        meta['logic'] = LogicProcess()

        return chat_history, meta


    msg.submit(respond, [msg, chatbot, meta], [msg, chatbot, meta])
    clear_history.click(clear_data, inputs = [meta],outputs = [chatbot, meta])

    demo.launch(server_port=7860)





