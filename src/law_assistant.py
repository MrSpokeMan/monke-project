from openai import OpenAI
from vector_db import VectorDB
import ollama
from cross_encoder import CrossEncoder

class LawAssistant:
    def __init__(self, vector_db: VectorDB = None, cross_encoder: CrossEncoder = None):
        self.client = ollama.Client()
        self.openai_client = OpenAI()
        self.db = vector_db if vector_db else VectorDB()
        self.x_encoder = cross_encoder if cross_encoder else CrossEncoder(self.db)
        self.messages = []

    def generate_response_research(self, query: str, reranker: bool = True):
        """
        Generates a response for a given query using the vector database and cross-encoder.
        :param query: The user's query.
        :param reranker: Whether to use the cross-encoder for reranking results.
        :return: A formatted response string.
        """
        if reranker:
            response, formated = self.db.get_response(query, search_width=50)
            _, formated = self.x_encoder.answer_query(query, response)
        else:
            _, formated = self.db.get_response(query)
        response = self.openai_client.chat.completions.create(
            model="gpt-4.1-nano-2025-04-14",
            messages=[
                {
                    "role": "user",
                    "content": formated
                }
            ]
        )
        return response.choices[0].message.content

    def generate_response(self, user_input, reranker: bool = True):
        self.messages = [
                {"role": "user", "content": user_input}
        ]

        response = self.client.chat(
            model="llama3.2",
            messages=self.messages,
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
        self.messages.append(response["message"])

        # Check if the model decided to use the provided function

        if not response["message"].get("tool_calls"):
            print("The model didn't use the function. Its response was:")
            print(response["message"]["content"])
            return "The model didn't use the function. Its response was:" + response["message"]["content"]

        # Process function calls made by the model
        if response["message"].get("tool_calls"):
            available_functions = {
                "get_response": self.db.get_response
            }

            for tool in response["message"]["tool_calls"]:
                function_to_call = available_functions[tool["function"]["name"]]
                function_args = tool["function"]["arguments"]
                if reranker:
                    function_args["search_width"] = 50
                function_response, formated = function_to_call(**function_args)
                if reranker:
                    function_response, formated = self.x_encoder.answer_query(user_input, function_response)
                # Add function response to the conversation
                tool_message = {
                    "role": "tool",
                    "content": formated,
                }
                self.messages.append(tool_message)

        # Second API call: Get final response from the model
        final_response = self.client.chat(model="llama3.2", messages=self.messages)

        self.messages.append(final_response["message"])
        return final_response["message"]["content"]


if __name__ == '__main__':
    assistant = LawAssistant()
    resp = assistant.generate_response("What is the law about industry?")
    print(resp)