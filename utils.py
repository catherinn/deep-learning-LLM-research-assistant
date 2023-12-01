from logical_state import LogicProcess
import json
from typing import List, Optional, Tuple
import re
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry



def get_phrase(text: str) -> List[str]:

    '''
        Splits a text into sentences using punctuation as separators.
    Args:
        text: The input text to be split into sentences.
    Returns:
        List[str]: A list of sentences extracted from the text.

    '''

    result = re.split(r'[.!?\n]', text)
    result = [phrase.strip() for phrase in result if phrase.strip()]
    return result


def text_is_value(text: str) -> bool:

    '''

        This function assesses the value of a text based on certain criteria.
        It returns True if the text is deemed valuable and False if not. The criteria include checking if the text is
        sufficiently long (more than 4 words) and whether it contains specific words that may indicate less valuable
        content, such as 'cookie,' 'copyright,' or 'date de mise en ligne.'
    Args:
        text:The text to be evaluated.

    Returns:
        True if the text is considered valuable, False if not.

    '''
    # False if short phrase
    text = text.lower()
    if len(text.split() ) <= 4:
        return False

    # False if containing specific words
    filters = ['cookie', 'copyright', 'date de mise en ligne']
    test_filtre = [True if re.search(filtre, text) else False for filtre in filters]
    pass
    if any(test_filtre):
        return False
    return True



if __name__ == '__main__':
    with open('https___www_pasdecalais_fr_Mobilite_rec_2.jsonl', 'r') as f:
        for index, line in enumerate(f):

            data = json.loads(line)
            if index == 1:
                break
    text = list(data.values())[0]

    result = list()
    for phrase in get_phrase(text):
        if text_is_value(phrase):
            result.append(phrase)

    print(result)

def check_url(url: str, logic:LogicProcess) -> Tuple[bool, str, LogicProcess]:

    if 'http://' not in url and 'https://' not in url:
        return False, 'Need https:// adress', logic

    session = requests.Session()
    retry = Retry(connect=1, backoff_factor=0.5, total=1)

    # remove the /n at the end may return 404 error
    url_traite = re.sub(r'\s+', '', url)

    adapter = HTTPAdapter(max_retries=retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    try:
        web_page = session.get(url_traite, verify=False)  # not check the certificates because can generate some error
    except requests.exceptions.ConnectionError as e:
        return False, 'Connection error', logic

    if web_page.status_code != 200:
        return False, str(web_page.status_code), logic
    else:
        logic.step = 1
        logic.url = url_traite
        return True, str(web_page.status_code), logic
