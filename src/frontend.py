import streamlit as st

import law_assistant


class LawBot:
    def __init__(self):
        self.bot = law_assistant.LawAssistant()

    def set_front(self):
        st.title("Ask LawBOT")

        # Database Update Section in Sidebar
        with st.sidebar:
            st.markdown("Update Database")
            st.markdown("This process may take some time..._")
            if st.button("Update Now"):
                with st.spinner("Updating the database... This may take some time."):
                    self.bot.db()
                st.success("Database update completed successfully!")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            st.chat_message(message["role"]).markdown(message["message"])

        prompt = st.chat_input("Pass your message to LawBOT")

        if prompt:
            st.chat_message("user").markdown(prompt)
            st.session_state.messages.append({"role": "user", "message": prompt})

            resp = self.bot.generate_response(prompt)
            st.chat_message("bot").markdown(resp)
            st.session_state.messages.append({"role": "bot", "message": resp})


if __name__ == "__main__":
    law_bot = LawBot()
    law_bot.set_front()
