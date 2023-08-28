"""
Author : Anuj Maurya
Description: FileChatBot allow to ask question regarding files.
Version : 1.0
Date: 28-08-2023
Azure Ticket Link : https://dev.azure.com/Generative-AI-Training/GenerativeAI/_workitems/edit/39/

"""

import os
import openai
import pinecone
from langchain.vectorstores import Pinecone
from dotenv import load_dotenv, find_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Environment Variables
load_dotenv(find_dotenv())

class FilesChatBot:
    def __init__(self, folder_path, similarity_search_size, chat_history_size, index_name):
        self.folder_path = folder_path
        self.similarity_search_size = similarity_search_size
        self.chat_history_size = chat_history_size
        self.index_name = index_name
        self.configure_api()
        self.components_initialize()

    # Set up OpenAI API configuration
    def configure_api(self):
        openai.api_type= os.getenv('API_TYPE')
        openai.api_key = os.getenv('OPENAI_API_KEY')
        openai.api_base = os.getenv('OPENAI_API_BASE')
        openai.api_version= os.getenv("API_VERSION")
    
    def components_initialize(self):
        # Initialize Pin Cone
        pinecone.init(
            api_key=os.getenv("PINECONE_API_KEY"),  # find at app.pinecone.io
            environment=os.getenv("PINECONE_ENV"),  # next to api key in console
        )

        self.vectorstore = self.get_vector_db()
        self.chat = self.get_conversation_chain()


    # Load File and Extract Raw Text
    def get_file_data(self):
        loader = DirectoryLoader(folder_path, glob='**/*.csv', loader_cls=CSVLoader)
        files_data = loader.load()

        loader = DirectoryLoader(folder_path, glob='**/*.pdf', loader_cls=PyPDFLoader)
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
    
    
    def get_vector_db(self):
        file_chunks = self.get_text_chunks()

        # all-MiniLM-L6-v2(DIMENSION): 384
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        
        # First, check if our index already exists. If it doesn't, we create it
        if self.index_name not in pinecone.list_indexes():
            # we create a new index
            pinecone.create_index(
                name=self.index_name,
                metric='cosine',
                dimension=384  
            )
        
            Pinecone.from_documents(file_chunks, embeddings, index_name=self.index_name)
        
        return Pinecone.from_existing_index(self.index_name, embedding=embeddings)
            


    # Creating Conversation cain
    def get_conversation_chain(self):
        llm = ChatOpenAI(temperature=0.0, model_kwargs={"engine": "GPT3-5"})

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
        result = self.chat({"question": question})
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
            question = input('Ask a question about your documents (or type "exit" to quit): ')
            if(question.lower() == 'exit'): break

            result = self.get_result(question)
            self.print_result(result)





# Driving Code
if __name__ == '__main__':
    """
    Question Examples: 
        1. what is the horsepower of Passport SPORT?
        2. what is the engine displacement of Civic LX?
        3. what's the joining date?
        4. What will be the working hours?
    """

    # Path to Folder
    folder_path = "./input"
    
    # Create a FileChatBot instance
    chat_bot = FilesChatBot(folder_path, 
                            similarity_search_size=7, 
                            chat_history_size=10,
                            index_name='files-chat-bot') # if not getting desired result, try increasing similarity_search_size

    # Starting the chat bot
    chat_bot.start_chat()
    