from ollama import Client
from modules.config import Config

from bs4 import BeautifulSoup
import re


def query_ollama(prompt, model=Config.MODEL):
    client = Client(host=Config.OLLAMA_API_BASE_URL)
    
    response = client.chat(
        model=model, 
        messages=[{'role': 'user', 'content': prompt}]
    )
    return response['message']['content']