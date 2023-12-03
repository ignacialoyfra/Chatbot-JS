
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
import torch
from sklearn.metrics.pairwise import cosine_distances
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
import os
import warnings
warnings.filterwarnings("ignore")

load_dotenv() 
api_key = os.getenv('OPENAI_API_KEY')
chat = OpenAI(temperature=0.4, model_name='gpt-3.5-turbo-16k', api_key=api_key)

def create_embeddings(texts):
    model_name = 'bert-base-multilingual-cased'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)


    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors='pt')


    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)

   
    embeddings = embeddings.numpy()

    return embeddings


def function_text_splitter(document, chunk_size:int, chunk_overlap:int):
    data = document.load()
    text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = chunk_size,
    length_function = len,
    chunk_overlap = chunk_overlap
    )
    documents_result = text_splitter.split_documents(data)
    return documents_result


def question_user(q:str):
  data_question = pd.DataFrame()
  emb = []
  q_list = []
  emb.append(create_embeddings(q))
  q_list.append(q)
  data_question["pregunta"] = q_list
  data_question["embedding_pregunta"] = emb
  return data_question



def data_metadata(data:object, data_p:object):
    data["distance_cos"] = data["embeddings_content"].apply(lambda x:cosine_distances(data_p.iloc[0,1],x[0].reshape(1,-1)))
    data = data.sort_values(by = "distance_cos",ascending=True)
    return data

def data_main(data:object, data_p:object):
    data = data_metadata(data, data_p)
    # data = filtered_data(param, data, "distance_cos")
    return data.iloc[0:10,:]

def documents_prompt(documents:list, pages:list):
    docs = []
    for doc in range(len(documents)):
        val_aux = documents[doc].metadata["page"]
        if val_aux in pages:
            docs.append(documents[doc])
        else:
            continue
    return docs

# Función que retorna una lista con las páginas que va a recibir el prompt
def get_pages(data:object, col_name:str):
    pages_content = []
    index_col = data.columns.get_loc(col_name)
    for index in range(data.shape[0]):
        val = data.iloc[index, index_col]
        pages_content.append(val["page"])
        pages_content = sorted(list(set(pages_content)))
    return pages_content

def function_main_documents(question:str, data:pd.DataFrame, documents:list):
    question = question_user(question)
    data_documents = data_main(data,question)
    pages = get_pages(data_documents, "metadata")
    docs_ = documents_prompt(documents, pages)
    return docs_, pages


def function_main_app(p:str, data:pd.DataFrame, documents:list):
    docs_, pages = function_main_documents(p, data, documents)
    # Definición del prompt
    template = """
        Eres Ali, un asistente virtual de personalidad extremadamente amable,tu proposito es ayudar con preguntas de JavaScript
        y debes responder solamente en base a la siguiente información: {context} para contestar la
        siguiente pregunta:{human_input} y tienes el siguiente contexto de la conversación {chat_history}.
        Responde solo en español destacando tu personalidad y utilizando solo la información que recibes.
        Contesta de forma detallada.
        Si no sabes la respuesta, sólo dilo."""
    prompt = PromptTemplate(
            input_variables=["chat_history", "human_input", "context"], template=template
            )
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="human_input")
                        # Se crea la cadena
    chain = load_qa_chain(
    chat, chain_type="stuff", memory=memory, prompt=prompt)
                        
    response = chain({"input_documents": docs_,
                        "human_input": p}, return_only_outputs=True)
    return response["output_text"], pages