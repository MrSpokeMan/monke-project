import streamlit as st
from openai import AsyncOpenAI
from pymilvus import MilvusClient

from cross_encoder import CrossEncoder
from embedding import EmbeddingModel
from law_assistant import LawAssistant
from settings import Settings
from vector_db import VectorDB


class LawBot:
    def __init__(self, assistant: LawAssistant):
        self.assistant = assistant

    def set_front(self):
        st.title("Ask LawBOT")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            st.chat_message(message["role"]).markdown(message["message"])

        prompt = st.chat_input("Pass your message to LawBOT")

        if prompt:
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "message": prompt})

            resp = self.assistant.generate_response_sync(prompt, use_reranker=False)
            st.chat_message("bot").markdown(resp)
            st.session_state.messages.append({"role": "bot", "message": resp})


if __name__ == "__main__":
    settings = Settings()

    openai_client = AsyncOpenAI(api_key=settings.openai_api_key)
    embedding_model = EmbeddingModel(
        settings.embedding_model,
    )
    milvus_client = MilvusClient(uri=settings.milvus_uri, token=settings.milvus_token)
    vector_db = VectorDB(embedding_model=embedding_model, milvus_client=milvus_client)
    cross_encoder = CrossEncoder(
        settings.cross_encoder_model,
    )
    law_assistant = LawAssistant(
        vector_db=vector_db,
        cross_encoder=cross_encoder,
        openai_client=openai_client,
        model_name=settings.llm_model,
    )
    law_bot = LawBot(law_assistant)
    law_bot.set_front()
