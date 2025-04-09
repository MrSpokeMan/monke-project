from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import vector_db


class LawAssistant:
    def __init__(self):
        self.llm = OllamaLLM(model="llama3.2")
        self.prompt = ChatPromptTemplate.from_template("""
        You are a legal assistant. Answer the question based on the provided laws.
        
        Laws: {docs}
        
        Question: {input}
        """)
        self.db = vector_db.VectorDB()

    def generate_response(self, user_input):
        resp = self.db.get_response(user_input)
        print(resp)
        # prompt = self.prompt.format_prompt(input=user_input)
        # response = self.llm(prompt)
        # return response


if __name__ == '__main__':
    assistant = LawAssistant()
    assistant.generate_response("What is the law about industry?")