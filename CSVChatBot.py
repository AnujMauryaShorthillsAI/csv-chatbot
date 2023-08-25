"""
Author : Anuj Maurya
Description: FileChatBot allow to ask question regarding a CSV file.
Version : 1.0
Date: 24-08-2023
Azure Ticket Link : https://dev.azure.com/Generative-AI-Training/GenerativeAI/_workitems/edit/39/

"""

import os
import openai
from dotenv import load_dotenv, find_dotenv
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

# Environment Variables
load_dotenv(find_dotenv())

class CSVChatBot:
    def __init__(self, file_path):
        self.file_path = file_path
        self.configure_api()
        self.components_initialize()

    # Set up OpenAI API configuration
    def configure_api(self):
        openai.api_type="azure"
        openai.api_key = os.getenv('OPENAI_API_KEY')
        openai.api_base = os.getenv('OPENAI_API_BASE')
        openai.api_version='2023-05-15'
    
    def components_initialize(self):
        self.vectorstore = self.get_vector_db()
        self.chat = self.get_conversation_chain()

    # Load File and Extract Raw Text
    def get_file_data(self):
        loader = CSVLoader(file_path=self.file_path)
        file_data = loader.load()

        return file_data
    
    def get_text_chunks(self):
        file_data = self.get_file_data()

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators='\n',
            length_function=len
        )

        return text_splitter.split_documents(file_data)
    
    
    # Saving and Loading vector db
    def get_vector_db(self):
        file_chunks = self.get_text_chunks()

        # all-MiniLM-L6-v2(DIMENSION): 384
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        try:
            vectorstore = FAISS.load_local("faiss_index", embeddings)
        except:
            vectorstore = FAISS.from_documents(documents=file_chunks, embedding=embeddings)

            # Save vector store
            vectorstore.save_local("faiss_index")
        
        return vectorstore
    

    # Creating Conversation cain
    def get_conversation_chain(self):
        llm = ChatOpenAI(temperature=0.0, model_kwargs={"engine": "GPT3-5"})
        memory = ConversationBufferWindowMemory(memory_key="chat_history", k=10, input_key='question', output_key='answer', return_messages=True)
        chat = ConversationalRetrievalChain.from_llm(llm=llm, retriever=self.vectorstore.as_retriever(), memory=memory, return_source_documents=True)

        return chat
    
    def get_result(self, question):
        result = self.chat({"question": question})
        return result
    
    def print_result(self, result):
        print("\n######### RESULT DETAILS ###########")
        print("Result: " + result['answer'], end="\n"*2)

        print("######### CHUNKS DETAILS ###########")
        for index, source in enumerate(result['source_documents']):
            print(f"######### {index+1} CHUNK DETAILS ###########")
            # print(f"{index} Chunk Detail content:")
            print("Chunk Content: ", source.page_content, end='\n'*2)
            print("Source File: ", source.metadata['source'],  end='\n'*2)
            print("Row Number: ", source.metadata['row'], end="\n"*2)

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
    """

    # Path to File
    file_path = "./input/HondaCANACompleteData.csv"
    
    # Create a FileChatBot instance
    chat_bot = CSVChatBot(file_path)

    # Starting the chat bot
    chat_bot.start_chat()
    