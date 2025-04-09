import streamlit as st
import vector_db
import law_assistant

class LawBot:
    def __init__(self):
        self.db  = vector_db.VectorDB()
        self.bot = law_assistant.LawAssistant()

    def set_front(self):
        st.title('Ask LawBOT')

        if 'messages' not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            st.chat_message(message['role']).markdown(message['message'])

        prompt = st.chat_input("Pass your message to LawBOT")

        if prompt:
            st.chat_message('user').markdown(prompt)
            st.session_state.messages.append({'role': 'user', 'message': prompt})

            resp = self.bot.generate_response(prompt)
            st.chat_message('bot').markdown(resp)
            st.session_state.messages.append({'role': 'bot', 'message': resp})


if __name__ == '__main__':
    law_bot = LawBot()
    law_bot.set_front()
