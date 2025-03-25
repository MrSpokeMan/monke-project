from openai import OpenAI
import fitz
import numpy as np
import os

class EmbeddingModel():
    def __init__(self):
        self.client = OpenAI()
        
    def get_embedding(self, text):
        response = self.client.embeddings.create(model="text-embedding-3-small", input=text)
        return np.array(response.data[0].embedding)
    
    
def extract_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    doc.close()
    return text

if __name__ == '__main__':
    # print(os.environ.keys())
    # emb = EmbeddingModel()
    text = extract_from_pdf('./data/CELEX_32024R2803_EN_TXT.pdf')
    print(text)
    # emb.get_embedding(text)