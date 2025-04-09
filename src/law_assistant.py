from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import vector_db
import ollama


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
        client = ollama.Client()
        messages = [
                {"role": "user", "content": user_input}
        ]
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
        messages.append(response["message"])

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
                messages.append(
                    {
                        "role": "tool",
                        "content": function_response[0][0]['entity']['name'] + " " + function_response[0][0]['entity']['text'],
                    }
                )

        # Second API call: Get final response from the model
        final_response = client.chat(model="llama3.2", messages=messages)

        return final_response["message"]["content"]


if __name__ == '__main__':
    assistant = LawAssistant()
    assistant.generate_response("What is the law about industry?")