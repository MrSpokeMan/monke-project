from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import vector_db
import ollama
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

class LawAssistant:
    def __init__(self):
        self.llm = OllamaLLM(model="llama3.2")
        # self.memory = ConversationBufferMemory(
        #     memory_key="history",
        #     input_key="input",  # only "input" is used for memory
        #     return_messages=True
        # )
        self.prompt = ChatPromptTemplate.from_template("""
        You are a legal assistant. Answer the question based on the provided laws.
        
        Laws: {docs}
        
        Question: {input}
        """)

        # self.chain = LLMChain(llm=self.llm, prompt=self.prompt, memory=self.memory)
        self.db = vector_db.VectorDB()
        self.conversation_history = []

    def generate_response(self, user_input):
        client = ollama.Client()
        messages = [
                {"role": "user", "content": user_input}
        ]

        context = self.conversation_history[-5:]

        messages = context.copy()

        response = client.chat(
            model="llama3.2",
            messages=messages,
            tools=[
                {
                    "type": "function",
                    "function": {
                        "name": "get_response",
                        "description": "Search about industry law in a vector database",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "prompt": {
                                    "type": "string",
                                    "description": "The search query",
                                },
                            },
                            "required": ["prompt"],
                        },
                    }
                }
            ]
        )

        # Add the model's response to the conversation history
        self.conversation_history.append(response["message"])

        # Check if the model decided to use the provided function

        if not response["message"].get("tool_calls"):
            print("The model didn't use the function. Its response was:")
            print(response["message"]["content"])
            return

        # Process function calls made by the model
        if response["message"].get("tool_calls"):
            available_functions = {
                "get_response": self.db.get_response,
            }

            for tool in response["message"]["tool_calls"]:
                function_to_call = available_functions[tool["function"]["name"]]
                function_args = tool["function"]["arguments"]
                function_response = function_to_call(**function_args)
                # Add function response to the conversation
                tool_message = {
                    "role": "tool",
                    "content": function_response[0][0]['entity']['name']
                               + " " + function_response[0][0]['entity']['text'],
                }
                self.conversation_history.append(tool_message)

        # Second API call: Get final response from the model
        final_context = self.conversation_history[-5:]
        final_response = client.chat(model="llama3.2", messages=final_context)

        self.conversation_history.append(final_response["message"])
        return final_response["message"]["content"]
    # def generate_response(self, user_input):
    #     # Get law information from your vector DB.
    #     try:
    #         docs_result = self.db.get_response(prompt=user_input)
    #     except Exception as e:
    #         docs_result = None
    #         print(f"Vector search error: {e}")
    #
    #     if docs_result:
    #         # Example assumption: docs_result contains a document in the structure below.
    #         doc_data = docs_result[0][0]['entity']
    #         doc_str = f"{doc_data['name']}: {doc_data['text']}"
    #     else:
    #         doc_str = "No relevant laws found."
    #
    #     # Run the chain with the full prompt: conversation history, law docs, and the user's question.
    #     response = self.chain.run(input=user_input, docs=doc_str)
    #     return response


if __name__ == '__main__':
    assistant = LawAssistant()
    assistant.generate_response("What is the law about industry?")