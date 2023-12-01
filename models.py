import logging
from langchain.chains import RetrievalQA
import torch
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from time import time
from datasets import load_dataset
import locale
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from loguru import logger
from langchain.embeddings import HuggingFaceEmbeddings
#from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import text_splitter
from langchain.vectorstores import FAISS
# load data from hugging face
from datasets import load_dataset
from torch import cuda, bfloat16
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM ,BitsAndBytesConfig, AutoConfig
from transformers import BitsAndBytesConfig
from typing import List, Optional
from langchain.docstore.document import Document
from langchain.llms import OpenAI
from pathlib import Path
import openai
import os
# Global configuration

if Path('open_ai_key.txt').exists():
    os.environ['OPENAI_API_KEY'] = open(Path('open_ai_key.txt'), 'r').read()
    logger.info('Open AI key loaded')
else:
    logger.warning('No open_ai_key.txt to load the key.')




# device = f'cuda:{cuda.current_device()}' if cuda.is_available() else 'cpu'
device ='cpu'
model_name_llm = 'gpt-3.5-turbo'
model_embedding_name = 'sentence-transformers/all-mpnet-base-v2'

logger.info(f'Device: {device}')
logger.info(f'Model llm: {model_name_llm}')
logger.info(f'Model embedding: {model_embedding_name}')

# Set the model on the target
torch.set_default_device(device)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=bfloat16
)

class LLM():
    def __init__(self):
        logger.info('Loading the LLM and the tokenizer')
        self.temperature = 1
        logger.info(f'Temperature: {self.temperature}')
        # self.llm, self.tokenizer = self.load_llm()

    def load_llm(self):
        pass
        # llm = AutoModelForCausalLM.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
        # tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-1_5", trust_remote_code=True)
        # return llm, tokenizer

    def answer_from_question(self, question: str, context: Optional[str|None] = None):
        # if context is None:
        #     inputs = self.tokenizer(f'can you answer to this question: {question}',
        #                             return_tensors="pt",
        #                             return_attention_mask=False)
        # else:
        #     inputs = self.tokenizer(f'make a summary of {context}', return_tensors="pt",
        #                    return_attention_mask=False)
        #
        # outputs = self.llm.generate(**inputs, max_length=400)
        # txt_output = self.tokenizer.batch_decode(outputs)
        #
        # # Extracting answer from the llm answer
        # shape_answer = r"Answer: (.*)"
        # answer = re.search(shape_answer, txt_output)

        messages = [
            {"role": "user", "content": f"Considering only this context: {context}, answer the question: {question} in less than 100 words"}
        ]
        rep = openai.ChatCompletion.create(model=model_name_llm,
                                           temperature=self.temperature,
                                           messages=messages,
                                           api_key=open(Path('open_ai_key.txt'), 'r').read())
        return rep["choices"][0]["message"]["content"]

        # if answer:
        #     answer = answer.group(1)
        #     return answer
        # else:
        #     return 'Shape of the answer does not contain "Answer:"'



class MyVectorStore():
    '''
    Class to load and make vector prediction from a corpus of text
    '''

    def __init__(self, dir_save: str, dir_load: Optional[str | None] = None):
        '''

        :param dir: dir from where save and load the model
        :param dir_load: dir containing text files to vectorize if no use scientific_papers dataset
        :return:
        '''

        if dir_load:
            if not Path(dir_load).exists():
                logger.warning(f'{dir_load} is not correct')
                return
        if Path(dir_save).exists():
            if not Path(dir_save).parent.exists():
                logger.warning(f'{dir_save} is not correct')
                return
            else: Path(dir_save).mkdir()

        self.dir = Path(dir_save)
        model_kwargs = {"device": device}
        self.embeddings = HuggingFaceEmbeddings(model_name=model_embedding_name, model_kwargs=model_kwargs, encode_kwargs={"batch_size": 1})
        self.vectorstore = self.load_vector_store(dir=dir_load)
        self.llm = LLM()

    def load_vector_store(self, dir: Optional[str | None] = None):

        '''
        Loads a sentence-transformers model and processes a dataset of scientific papers to create and store dense vector embeddings
        for abstracts. It then allows querying the stored vectors for semantic similarity.
        dir: [optional] path to a dir of files.txt to vectorize if no use dataset scientific_papers
        Returns:
        FAISSVectorStore: A vector store containing the embeddings of abstracts from scientific papers.
        '''

        # Global configuration of thr embedding model
        '''
        This is a sentence-transformers model: It maps sentences & paragraphs to a 768 dimensional dense vector space and can
         be used for tasks like clustering or semantic search.
        '''
        locale.getpreferredencoding = lambda: "UTF-8"
        logger.info('Loading the documents')
        if dir is None:
            # Load model from HuggingFace Hub
            dataset = load_dataset("scientific_papers", 'pubmed')
            # articles = dataset['train'][0:10]['article']
            abstracts = dataset['train'][0:10]['abstract']
            # section_names = dataset['train'][0:10]['section_names']
        else:
            dir = Path(dir)
            file_texts = dir.glob('*.txt')
            abstracts = [open(file_text, 'r').read() for file_text in tqdm(file_texts, desc='Loading text extracted from web site.')]

        # A loop that allows increasing the chunk size while preserving the separation between the abstracts.
        splits: List[Document] = []
        chunk_size = 500
        logger.info(f'chunk_size: {chunk_size}')
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=20)
        for i, abstract in tqdm(enumerate(abstracts), desc='Splitting text files'):
            splits += text_splitter.create_documents([abstract])


        # Store embeddings in the vector store
        logger.info('Loading vectores from documents')
        vectorstore = FAISS.from_documents(splits, self.embeddings)
        logger.info('Loading vectores from documents is completed')

        return vectorstore

    def get_closet_texts_from_vectorstore(self, text:str, k: Optional[int] = 2) -> [List, str]:

        # The results are saved and loaded into a context
        list_result = self.vectorstore.similarity_search_with_score(text, k = k)
        results = list()
        scores = list()
        for index, (text, score) in enumerate(list_result):
            results.append(text.page_content)
            scores.append(score)

        return scores, "\n\n".join([str(result) for result in results])


    def save(self):

        '''Persist the vectors locally on disk'''
        self.vectorstore.save_local(self.dir / "faiss_index_constitution")

    def load(self):
        self.vectorstore = FAISS.load_local(self.dir / "faiss_index_constitution", self.embeddings)

    def answer_from_vectorstore(self, question: str, context: str) -> str:
        return self.llm.answer_from_question(question=question, context=context)
