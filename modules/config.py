import os

base_url = os.environ.get('OLLAMA_API_BASE_URL', "http://127.0.0.1:11434/api")
if base_url.endswith('/'):
    base_url = base_url.rstrip('/')


class Config:
    DB_PATH = os.path.join(os.getcwd(), "chromadb")
    MODEL = os.environ.get('MODEL', "llama3")
    EMBEDDING_MODEL_NAME = os.environ.get('EMBEDDING_MODEL_NAME', "all-MiniLM-L6-v2") # "all-MiniLM-L6-v2", "mxbai-embed-large"
    OLLAMA_API_BASE_URL = base_url
    HUGGING_FACE_EMBEDDINGS_DEVICE_TYPE = os.environ.get('HUGGING_FACE_EMBEDDINGS_DEVICE_TYPE',
                                                         "cpu")
