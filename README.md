# Research assistant

## The task: NLP domain

## Goal 
The goal of this project is to create a prototype of a solution that will inform the users on the state of the art of scientific research on a topic of the user's interest. We achieve this by enabling the user to conduct searches on any desired website through a chatbot-style interface. 


## Warning
Extracting information from websites without their explicit permission can be against the terms of use of many websites. Before undertaking any such action, make sure to adhere to the policies of the respective website.

## Noteworthy tools used:
- HuggingFace Datasets for loading the initial dataset for vector search
- HuggingFaceEmbeddings, RecursiveCharacterSplitter and FAISS vector store for correctly embedding and tokenizing our dataset and FAISS similarity search for finding 
the right context to be used in chat
- Gradio for prototyping our work as a chat interface
- Langchain libraries: FAISS vectorstore, RecursiveCharacterTextSplitter
- Beautiful Soup library for web scraping 
- Docker for containerizing our work

We also experimented with the following (see DeepLearningProject - experiments.ipynb file):
- Langchain libraries: tools, chains, agents, PromptTemplates
- BitsAndBytesConfig for optimizing our LLM's memory efficiency and speed
- using the APIs that pubmed and arxiv provide and use them as retrievers


## Make test

Write your OpenAI key into the empty open_ai_key.txt text file.


Make sure you have installed Docker and that it is currently running.
```python
docker build -t deep_learning_project:0 .

docker run -p 7860:7860 -e GRADIO_SERVER_NAME=0.0.0.0 --name deep_learning_project_test deep_learning_project:0
```
Redirect yourself to your web browser and enter the address: http://127.0.0.1:7860/
You can start interacting with the chatbot.


Attention, if you stop the Docker and run it again to enter a new URL, make sure to close all Gradio windows. Otherwise, it will reload the variables from the previous window (and thus continue searching on the old URL).
