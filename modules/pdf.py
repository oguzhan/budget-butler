import os
import time
import uuid
import json
import re
from pathlib import Path

from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from modules.config import Config
from modules.ollama import query_ollama
from PyPDF2 import PdfReader 

class PDFHelper:
    def __init__(self, model_name: str = Config.MODEL,
                 embedding_model_name: str = Config.EMBEDDING_MODEL_NAME):
        self._model_name = model_name
        self._embedding_model_name = embedding_model_name

    def extract_text(self, pdf_file_path: str) -> str:
        """
        Extract text from a PDF file using PyPDF2.
        """
        text = ""
        try:
            with open(pdf_file_path, 'rb') as file:
                pdf_reader = PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
        except Exception as e:
            print(f"Error extracting text from PDF: {str(e)}")
        return text

    def ask(self, pdf_file_path: str, question: str) -> str:
        vector_store_directory = os.path.join(str(Path.home()), 'langchain-store', 'vectorstore', 'pdf-doc-helper-store', str(uuid.uuid4()))
        os.makedirs(vector_store_directory, exist_ok=True)
        print(f"Using vector store: {vector_store_directory}")

        # Load the Embedding Model
        embed = self._load_embedding_model(model_name=self._embedding_model_name)

        # Load and split the documents
        docs = self._load_pdf_data(file_path=pdf_file_path)
        documents = self._split_docs(documents=docs)

        # Create vectorstore
        vectorstore = self._create_embeddings(chunks=documents, embedding_model=embed, storing_path=vector_store_directory)

        # Convert vectorstore to a retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

        # Retrieve relevant documents
        relevant_docs = retriever.get_relevant_documents(question)

        # Combine relevant documents into a single context
        context = "\n\n".join([doc.page_content for doc in relevant_docs])

        # Create the prompt
        prompt = f"""
        Use the following pieces of context to answer the question at the end. 
        If you don't know the answer, just say that you don't know, don't try to make up an answer.

        Context:
        {context}

        Question: {question}

        Answer:
        """

        # Get response from Ollama
        response = query_ollama(prompt, model=self._model_name)

        return response.strip()

    def _load_pdf_data(self, file_path):
        loader = PyMuPDFLoader(file_path=file_path)
        docs = loader.load()
        return docs

    def _split_docs(self, documents, chunk_size=1000, chunk_overlap=20):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = text_splitter.split_documents(documents=documents)
        return chunks

    def _load_embedding_model(self, model_name, normalize_embedding=True):
        print("Loading embedding model...")
        start_time = time.time()
        hugging_face_embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': Config.HUGGING_FACE_EMBEDDINGS_DEVICE_TYPE},
            encode_kwargs={
                'normalize_embeddings': normalize_embedding
            }
        )
        end_time = time.time()
        time_taken = round(end_time - start_time, 2)
        print(f"Embedding model load time: {time_taken} seconds.\n")
        return hugging_face_embeddings

    def _create_embeddings(self, chunks, embedding_model, storing_path="vectorstore"):
        print("Creating embeddings...")
        e_start_time = time.time()

        # Create the embeddings using FAISS
        vectorstore = FAISS.from_documents(chunks, embedding_model)

        e_end_time = time.time()
        e_time_taken = round(e_end_time - e_start_time, 2)
        print(f"Embeddings creation time: {e_time_taken} seconds.\n")

        print("Writing vectorstore..")
        v_start_time = time.time()

        # Save the model in a directory
        vectorstore.save_local(storing_path)

        v_end_time = time.time()
        v_time_taken = round(v_end_time - v_start_time, 2)
        print(f"Vectorstore write time: {v_time_taken} seconds.\n")

        # return the vectorstore
        return vectorstore
    
    def extract_transactions(self, pdf_file_path: str) -> list:
        full_text = self.extract_text(pdf_file_path)
        
        if not full_text.strip():
            print("No text extracted from PDF.")
            return []
        
        prompt = f"""
        Extract all transactions from the following text. 
        For each transaction, provide the date (in YYYY-MM-DD format), amount (in €), and description.
        Format the response as a list of dictionaries in JSON format.
        If no transactions are found, return an empty list.

        Text:
        {full_text}

        Example output format:
        [
            {{"date": "2024-06-22", "amount": "-4.30", "description": "TWITTER PAID FEATURES"}},
            {{"date": "2024-06-23", "amount": "-9.52", "description": "PROTON"}}
        ]
        """
        
        response = query_ollama(prompt, model=self._model_name)
        
        try:
            # Clean up the response to ensure valid JSON
            cleaned_response = re.sub(r'(\d),(\d)', r'\1\2', response)
            transactions = json.loads(cleaned_response)
            
            if isinstance(transactions, list):
                # Clean up amount values
                # for transaction in transactions:
                #     transaction['amount'] = transaction['amount'].replace(',', '')
                return transactions
            else:
                print(f"Unexpected response format: {cleaned_response}")
                return []
        except json.JSONDecodeError:
            print(f"Error decoding JSON: {response}")
            # Attempt to extract transactions using regex if JSON parsing fails
            return self.extract_transactions_fallback(response)


    def extract_transactions_fallback(self, response: str) -> list:
        pattern = r'\{\s*"date":\s*"([\d-]+)",\s*"amount":\s*"([^"]+)",\s*"description":\s*"([^"]+)"\s*\}'
        matches = re.findall(pattern, response)
        transactions = []
        for match in matches:
            date, amount, description = match
            amount = amount.replace(',', '')
            transactions.append({
                "date": date,
                "amount": amount,
                "description": description
            })
        return transactions
        
__all__ = ['PDFHelper']