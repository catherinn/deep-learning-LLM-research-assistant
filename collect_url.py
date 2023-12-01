
from loguru import logger
from bs4 import BeautifulSoup as bs
import pickle
import shutil
from pathlib import Path
from urllib.parse import urljoin
from typing import List, Optional, Tuple
import re
from tqdm import tqdm
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from utils import get_phrase, text_is_value
from datetime import datetime

def get_txt_from_url(url: str, balise: List[str] = ['p', 'h1', 'h2', 'h3', 'ul', 'ol', 'li']) -> tuple[str, str]:

    '''
        Retrieves the content of a web page, extracts text inside the specified HTML tag, and cleans it.
    Args:
        url: The URL of the web page to scrape.
        balise: The HTML tag whose text needs to be extracted.

    Returns:
        tuple: A tuple containing the URL and the cleaned text of the page.
            - The first element is the URL of the web page.
            - The second element is the cleaned text.
    '''


    # This part is to not trigger error by requesting a server to many time and to quickly
    session = requests.Session()
    retry = Retry(connect = 1, backoff_factor = 0.5, total = 1)

    # remove the /n at the end may return 404 error
    url_traite = re.sub(r'\s+', '', url)

    adapter = HTTPAdapter(max_retries = retry)
    session.mount('http://', adapter)
    session.mount('https://', adapter)
    web_page = session.get(url_traite, verify=False)  # not check the certificates because can generate some error

    if web_page.status_code != 200:
        logger.warning(f"Received {web_page.status_code} status code for the URL: {url}")
        return url, ''

    html = web_page.content
    soup = bs(html, 'html.parser')
    # soup.find(balise) # get vthe text inside the specific balises
    text_clean = soup.get_text(separator=' ', strip=True) #re.sub(r'(\n\s*)+', '\n ', soup.text)

    return url, text_clean
    # ic(xml_content)




def clean_text(text: str):
    pass


def get_urls(url: str,
            filtre_in: List[str],
            filtre_not_in: List[str] = None,
            main: Optional[bool]=True,
            level:Optional[int]=1,
            exploration: Optional[List[str]] = None) -> Tuple[list, list]:

    '''
    Web Page Scraping and Link Exploration

    This code is designed to scrape web pages and explore their links recursively.
    It performs web scraping on the given URL and collects all unique links found in the page, subject to various filtering options.
    Writes the 'url' found in a generated file.

    Args:
    - url (str): The URL to start web scraping.
    - filtre_in (list): A list of keywords to include in URLs.
    - filtre_not_in (list): A list of keywords to exclude from URLs.
    - level (int): The maximum depth of recursive exploration.
    - main (bool): A flag to indicate whether this is the main call (generates output file).
    - exploration (list): A list of URLs already explored.

    Returns:
    - links (list): A list of unique URLs discovered during web scraping.
    - exploration (list): Updated list of explored URLs.

    '''

    if level < 0:
        return list(), exploration
    if exploration is None:
        exploration = [url] # Initialisation list of URL already explored, to decrease time during recursivity

    web_page = requests.get(url)
    link_buffer = list()
    # Vérifier si la requête a réussi
    if web_page.status_code == 200:
        soup = bs(web_page.text, 'html.parser')
        links = soup.find_all('a')

        for link in tqdm(links, desc= f'collect URL, level: {level}'):
            href = link.get('href')
            full_url = urljoin(url, href).lower()
            if full_url not in exploration: # If the URL not check yet, implement the exploration and add it to the list of exploration
                exploration.append(full_url)
                if re.search('http', full_url):  # check the link is in http or https
                    # if filter is implemented :
                    if filtre_in:
                        # is there matching with the list of filter ? + dont add this kind of link http://telmedia.fr#btn-menu
                        is_true = [True if (re.search(mot.lower(), full_url) != None and '#' not in full_url)
                                   else False for mot in filtre_in]
                        if any(is_true):
                            # recursif call
                            sub_link, exploration = get_urls(url = full_url, filtre_in = filtre_in, filtre_not_in = filtre_not_in, level = level - 1, main = False, exploration = exploration)
                            if not is_containing_forbidden_word(full_url, filtre_not_in): #remove useless url according filtre_not_in
                                link_buffer.append(full_url) # final url we add in the function
                            link_buffer = link_buffer + sub_link
        links = list(set(link_buffer))

        # Writting the  results
        if main:
            # removes special characters from 'url' and uses it to generate a file name.
            name_file = re.sub(r'[!@#$%^&*()_+[\]{}|;:\',.<>?~`/]', '_', url)
            with open(f'{name_file}_rec_{level}.txt', "w") as f:
                # Écrit chaque élément de la liste suivi d'un saut de ligne
                f.write(url + "\n")
                for link in links:
                    f.write(link + "\n")
            logger.info(f'get {len(links)} results.')
            logger.info(f'Filter  used {filtre_in} .')

        return links, exploration
    else:
        logger.warning(f"Received {web_page.status_code} status code for the URL: {url}")
        return list(),  exploration


def extact_txt_from_urls(url_library: List, dir_output: str):

    '''
    Extracts valuable phrases from websites and saves them in a JSONL (JSON Lines) file.

    This function reads an input text file line by line, where each line represents an URL.
    Collect the text from it, and  extracts valuable phrases from the text content, ensuring they are unique (no 2 times the same phrase)
    The extracted phrases are saved in dir_output : 1 txt file per URL, the dic[txt extracted] =  URL is saved also in pkl file in dir_output.

    Args:
        txt_file (str): The path to the input text file, which may contain URLs or text content.
        dir_output (str): The path to the output dir where will be saved the texts file and the dict.pkl.

    Returns:
        None
    '''

    logger.info('Extractions texts from URL -- started --')

    # Create dir output if exist create a mnew one with time stamp
    if not (Path(dir_output).parent).exists():
        logger.error(f'{dir_output} not valid.')
        return
    else:
        if Path(dir_output).exists():
            if 'del' in Path(dir_output).name:
                shutil.rmtree(dir_output)
                Path(dir_output).mkdir()
            else:
                timestamp = datetime.now().strftime(f"_%d_%m_%H_%M_%S")
                dir_output = dir_output + timestamp
                Path(dir_output).mkdir()
        else:
            Path(dir_output).mkdir()


    library_phrases = [] # library of phrases already extracted previously
    dict_text = {} # dict[text] = URL

    #
    # Read each URL
    for url_index, line in enumerate(url_library):
        # logger.info(f'URL: {line}')
        phrase_filtered = list()
        url, txt = get_txt_from_url(line) # Extract text from URL

        # Clean the text
        list_phrase = get_phrase(txt) # split the text in phrases
        for phrase in list_phrase: # keep only the valuable phrases
            if text_is_value(phrase) and phrase not in library_phrases:
                library_phrases.append(phrase)
                phrase_filtered.append(phrase)
        paragraphe = " ".join(phrase_filtered) # full cleaned text from URL
        name_txt_file = f'{str(url_index)}.txt'
        with open(Path(dir_output)/ f'{name_txt_file}', 'w') as fichier_txt:
            try:
                fichier_txt.write(paragraphe)
            except UnicodeEncodeError:
                logger.warning(f'Encodage error during extracting of this url: {url}')
        dict_text[name_txt_file] = url

    # Saving the dictionary
    with open(Path(dir_output)/ f'url.pkl', 'wb') as pkl_file:
        pickle.dump(dict_text, pkl_file)
    logger.info(f'Work completed and saved in {dir_output}.')


def is_containing_forbidden_word(text:str, words: List[str]) -> bool:
    '''
    This function checks if a given text contains any of the forbidden words provided in a list.
     It uses regular expressions to search for each word in the text and returns True if at least one forbidden word is found.
     It logs a message indicating the removed text when a forbidden word is found.
    Args:
        text: The text in which you want to check for forbidden words.
        words: A list of forbidden words to search for in the text.

    Returns:
    True if the text contains at least one forbidden word.
    False if none of the forbidden words are found in the text.
    '''
    for word in words:
        pattern = re.compile(word.lower())
        if re.search(pattern, text.lower()):
            logger.info(f'removed: {text}')
            return True
    return False


if __name__ == '__main__':

    # Collect valid url related to one and write all the results
    # get_urls('https://www.caf.fr/partenaires/', level = 1 , filtre_in = ['enfan', 'mater', 'parent'], filtre_not_in=['facebook', 'twitter', 'pdf'])

    # Get text from url text - file
    dir_output = Path.cwd() / 'library_caf'
    dir_txt = Path.cwd() / 'https___www_caf_fr_partenaires__rec_1.txt'
    extact_txt_from_urls(txt_file=str(dir_txt), dir_output=str(dir_output))
