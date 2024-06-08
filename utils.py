import chromadb
import numpy as np
import openai

class Embeddings:
    def __init__(self):
        self._client = chromadb.PersistentClient('embeddings')
        openai_ef = chromadb.utils.embedding_functions.OpenAIEmbeddingFunction(
                model_name="text-embedding-3-small"
            )
        self._ra_fa = Collection(self._client.get_collection(
                'ra_fa', 
                embedding_function=openai_ef))
        self._ra_fc = Collection(self._client.get_collection(
                'ra_fc', 
                embedding_function=openai_ef))
        self._rc_fa = Collection(self._client.get_collection(
                'rc_fa', 
                embedding_function=openai_ef))
        self._rc_fc = Collection(self._client.get_collection(
                'rc_fc', 
                embedding_function=openai_ef))
    
    @property
    def ra_fa(self):
        return self._ra_fa
    
    @property
    def ra_fc(self):
        return self._ra_fc
    
    @property
    def rc_fa(self):
        return self._rc_fa
    
    @property
    def rc_fc(self):
        return self._rc_fc
    

class Collection:
    def __init__(self, collection):
        self._collection = collection
        get_result = self._collection.get(include=['embeddings', 'documents'])
        self._embeddings = np.array(get_result['embeddings'])
        self._ids = np.array(get_result['ids'])
        self._documents = np.array(get_result['documents'])
    
    @property
    def count(self):
        return self._collection.count()
    
    def get_sample(self, n=3000):
        choice = np.random.choice(self._collection.count(), size=n, replace=False)        
        return self._ids[choice], self._embeddings[choice], self._documents[choice]
    
    def query(self, text, n=10):
        return self._collection.query(query_texts=[text], n_results=n, include=['documents'])
    
class LLM:
    def __init__(self):
        self._oai = openai.OpenAI()
    def message(self, content, sys_prompt=None):
        chat_completion = self._oai.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": sys_prompt or "You are a helpful assistant",
                },
                {
                    "role": "user",
                    "content": content,
                }
            ],
            model="gpt-3.5-turbo-0125",
        )
        return chat_completion.choices[0].message.content
        