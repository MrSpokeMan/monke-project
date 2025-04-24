import vector_db
import ollama

class LawAssistant:
    def __init__(self):
        self.client = ollama.Client()
        self.db = vector_db.VectorDB()
        self.messages = []

    def generate_response(self, user_input):
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
                "get_response": self.db.get_response,
            }

            for tool in response["message"]["tool_calls"]:
                function_to_call = available_functions[tool["function"]["name"]]
                function_args = tool["function"]["arguments"]
                function_response, format = function_to_call(**function_args)
                # Add function response to the conversation
                tool_message = {
                    "role": "tool",
                    "content": format,
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