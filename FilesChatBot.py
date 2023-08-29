"""
Author : Anuj Maurya
Description: FileChatBot allow to ask question regarding files.
Version : 1.0
Date: 24-08-2023
Azure Ticket Link : https://dev.azure.com/Generative-AI-Training/GenerativeAI/_workitems/edit/39/

"""

import os
import openai
from LLMUsage import LLMUsage
from helicone.openai_proxy import openai
from langchain.vectorstores import FAISS
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.callbacks import get_openai_callback
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Environment Variables
load_dotenv(find_dotenv())

class FilesChatBot:
    def __init__(self, folder_path, similarity_search_size, chat_history_size):
        self.folder_path = folder_path
        self.similarity_search_size = similarity_search_size
        self.chat_history_size = chat_history_size
        self.configure_api()
        self.components_initialize()

    # Set up OpenAI API configuration
    def configure_api(self):
        openai.api_type= os.getenv('API_TYPE')
        openai.api_key = os.getenv('OPENAI_API_KEY')
        openai.api_base = os.getenv('OPENAI_API_BASE')
        openai.api_version= os.getenv("API_VERSION")
    
    def components_initialize(self):
        self.vectorstore = self.get_vector_db()
        self.chat = self.get_conversation_chain()

    # Load File and Extract Raw Text
    def get_file_data(self):
        loader = DirectoryLoader(self.folder_path, glob='**/*.csv', loader_cls=CSVLoader)
        files_data = loader.load()

        loader = DirectoryLoader(self.folder_path, glob='**/*.pdf', loader_cls=PyPDFLoader)
        files_data += loader.load()
        
        return files_data
    
    def get_text_chunks(self):
        files_data = self.get_file_data()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators='\n',
            length_function=len
        )

        return text_splitter.split_documents(files_data)
    
    
    # Saving and Loading vector db
    def get_vector_db(self):
        file_chunks = self.get_text_chunks()

        # all-MiniLM-L6-v2(DIMENSION): 384
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # create, load and save vector
        try:
            vectorstore = FAISS.load_local("faiss_index", embeddings)
        except:
            vectorstore = FAISS.from_documents(documents=file_chunks, embedding=embeddings)

            # Save vector store
            vectorstore.save_local("faiss_index")

        # print(vectorstore.index_to_docstore_id)
        return vectorstore
    
    def __update_vector_db(self):
        file_chunks = self.get_text_chunks()

        # all-MiniLM-L6-v2(DIMENSION): 384
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        vectorstore = FAISS.from_documents(documents=file_chunks, embedding=embeddings)
        vectorstore.save_local("faiss_index")

        # Reinitialize the component
        self.components_initialize()

        print("Vector DB Updated Successfully.")
    

    # Creating Conversation cain
    def get_conversation_chain(self):
        llm = ChatOpenAI(temperature=0.0, 
                         model_kwargs={"engine": "GPT3-5"},
                         headers={
                             "Helicone-Auth": os.getenv('HELICONE_AUTH_KEY'),
                             "Helicone-User-Id": os.getenv('HELICONE_USER_ID')
                         })

        # Storing K Previous Chats
        memory = ConversationBufferWindowMemory(
            memory_key="chat_history", 
            k=self.chat_history_size, 
            input_key='question', 
            output_key='answer', 
            return_messages=True)

        # Retrieving of Top 7 Similarity Search
        chat = ConversationalRetrievalChain.from_llm(
            llm=llm, 
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": self.similarity_search_size}), 
            memory=memory, return_source_documents=True)

        return chat
    
    def get_result(self, question):
        with get_openai_callback() as cb:
            result = self.chat({"question": question})
            print(cb.total_tokens)
            LLMUsage.add_usage_details(cb)
        
        return result
    
    def print_result(self, result):
        print("######### CHUNKS DETAILS ###########")
        for index, source in enumerate(result['source_documents']):
            print(f"######### {index+1} CHUNK DETAILS ###########")
            print("Chunk Content: ", source.page_content, end='\n'*2)
            print("Source File: ", source.metadata['source'],  end='\n'*2)

            # Only CSV File Contains Row.
            if(source.metadata.get('row')):
                print("Row Number: ", source.metadata['row'], end="\n"*2)
        
        print("\n######### RESULT DETAILS ###########")
        print("Result: " + result['answer'], end="\n"*2)

    # Starting Q&A
    def start_chat(self):
        while(True):
            question = input('Ask a question about your documents (or type "exit" to quit / or type "update db" to load files): ')
            if(question.lower() == 'exit'): break
            if(question.lower() == 'update db'):
                accept = input('Warning: VectorDB will be updated with current input files. To Confirm Press "y": ')
                if(accept.lower() == 'y'):
                    self.__update_vector_db()
                continue
            result = self.get_result(question)
            self.print_result(result)





# Driving Code
if __name__ == '__main__':
    """
    Question Examples: 
        1. what is the horsepower of Passport SPORT?
        2. what is the engine displacement of Civic LX?
        3. What is the joining date mentioned in my offer letter?
        4. What will be the working hours?
    """

    # Path to Folder
    folder_path = "./input"
    
    # Create a FileChatBot instance
    chat_bot = FilesChatBot(folder_path, 4, 10)

    # Starting the chat bot
    chat_bot.start_chat()
    