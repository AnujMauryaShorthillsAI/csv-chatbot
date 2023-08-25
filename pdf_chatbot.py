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

class FileChatBot:
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
    def get_raw_text(self):
        loader = CSVLoader(file_path=self.file_path)
        file_data = loader.load()

        return file_data
    
    def get_text_chunks(self):
        file_data = self.get_raw_text()

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
        memory = ConversationBufferWindowMemory(memory_key="chat_history", k=10, return_messages=True)
        chat = ConversationalRetrievalChain.from_llm(llm=llm, retriever=self.vectorstore.as_retriever(), memory=memory)

        return chat
    
    def get_answer(self, question):
        result = self.chat({"question": question})
        return result['answer']
    
    # Starting Q&A
    def start_chat(self):
        while(True):
            question = input('Ask a question about your documents (or type "exit" to quit): ')
            if(question.lower() == 'exit'): break

            # Similar Chunks Metadata
            similary_chunks = self.vectorstore.similarity_search(question)
            for chunk in similary_chunks:
                print("Source File: " , chunk.metadata['source'])
                print("Row Number: " , chunk.metadata['row'], end="\n"*2)

            result = self.get_answer(question)
            print("Result: " + result, end="\n"*2)



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
    chat_bot = FileChatBot(file_path)

    # Starting the chat bot
    chat_bot.start_chat()
    