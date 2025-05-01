import ollama
from tqdm.auto import tqdm

class Evaluation:
    def __init__(self, context_list: list[str]):
        self.client = ollama.Client()
        self.context_list = context_list if context_list else []
        self.QA_GENERATION_PROMPT = """
        Your task is to write a factoid question and an answer given a context.
        Your factoid question should be answerable with a specific, concise piece of factual information from the context.
        Your factoid question should be formulated in the same style as questions users could ask in a search engine.
        This means that your factoid question MUST NOT mention something like "according to the passage" or "context".
        
        Provide your answer as follows:
        
        Output:::
        Factoid question: (your factoid question)
        Answer: (your answer to the factoid question)
        
        Now here is the context.
        
        Context: {context}\n
        Output:::"""
        self.QA_CRITIQUE_GROUNDEDNESS = """
        You will be given a context and a question.
        Your task is to provide a 'total rating' scoring how well one can answer the given question unambiguously with the given context.
        Give your answer on a scale of 1 to 5, where 1 means that the question is not answerable at all given the context, and 5 means that the question is clearly and unambiguously answerable with the context.

        Provide your answer as follows:

        Answer:::
        Evaluation: (your rationale for the rating, as a text)
        Total rating: (your rating, as a number between 1 and 5)

        You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

        Now here are the question and context.

        Question: {question}\n
        Context: {context}\n
        Answer::: """
        self.QA_CRITIQUE_RELEVANCE = """
        You will be given a question.
        Your task is to provide a 'total rating' representing how useful this question can be to machine learning developers building NLP applications with the Hugging Face ecosystem.
        Give your answer on a scale of 1 to 5, where 1 means that the question is not useful at all, and 5 means that the question is extremely useful.

        Provide your answer as follows:

        Answer:::
        Evaluation: (your rationale for the rating, as a text)
        Total rating: (your rating, as a number between 1 and 5)

        You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

        Now here is the question.

        Question: {question}\n
        Answer::: """
        self.QA_CRITIQUE_STANDALONE = """
        You will be given a question.
        Your task is to provide a 'total rating' representing how context-independent this question is.
        Give your answer on a scale of 1 to 5, where 1 means that the question depends on additional information to be understood, and 5 means that the question makes sense by itself.
        For instance, if the question refers to a particular setting, like 'in the context' or 'in the document', the rating must be 1.
        The questions can contain obscure technical nouns or acronyms like Gradio, Hub, Hugging Face or Space and still be a 5: it must simply be clear to an operator with access to documentation what the question is about.

        For instance, "What is the name of the checkpoint from which the ViT model is imported?" should receive a 1, since there is an implicit mention of a context, thus the question is not independent from the context.

        Provide your answer as follows:

        Answer:::
        Evaluation: (your rationale for the rating, as a text)
        Total rating: (your rating, as a number between 1 and 5)

        You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

        Now here is the question.

        Question: {question}\n
        Answer::: """

    def __call__(self):
        result = self._generate_questions()
        res = self._fill_dataset(result)
        print(res)

    def _call_llm(self, query: str):
        response = self.client.chat(
            model="llama3.2",
            messages=[
                {"role": "user", "content": query}
            ]
        )
        return response["message"]["content"]

    def _generate_questions(self):
        result_list = []
        for context in tqdm(self.context_list):
            result_dict = {"context": context}
            query = self.QA_GENERATION_PROMPT.format(context=context)
            response = self._call_llm(query)
            result_dict["question"] = response.split("Factoid question: ")[1].split("Answer: ")[0].strip()
            result_dict["answer"] = response.split("Answer: ")[1].strip()
            result_list.append(result_dict)
        return result_list

    def _fill_dataset(self, outputs):
        for output in tqdm(outputs):
            evaluations = {
                "groundedness": self._call_llm(self.QA_CRITIQUE_GROUNDEDNESS.format(question=output["question"], context=output["context"])),
                "relevance": self._call_llm(self.QA_CRITIQUE_RELEVANCE.format(question=output["question"])),
                "standalone": self._call_llm(self.QA_CRITIQUE_STANDALONE.format(question=output["question"]))
            }
            try:
                for criterion, evaluation in evaluations.items():
                    score, eval = (
                        int(evaluation.split("Total rating: ")[-1].strip()),
                        evaluation.split("Total rating: ")[-2].split("Evaluation: ")[1].strip(),
                    )
                    output.update({f"{criterion}_score": score, f"{criterion}_eval": eval})
            except Exception as e:
                print(f"Error processing evaluation: {e}")
        return outputs


if __name__ == '__main__':
    eval = Evaluation(context_list = ['This is a test context'])
    eval()