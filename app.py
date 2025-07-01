import streamlit as st
from dotenv import load_dotenv
import os

# --- UPDATED IMPORTS ---
# We now import the function to LOAD the pre-built index, not build it.
from src.rag_processor import load_vector_db
from src.llm_handler import create_effective_rag_chain
from src.utils import translate_text

# Load environment variables from .env file
load_dotenv()

# --- UI Configuration (This section remains unchanged) ---
UI_TEXT = {
    "en": {
        "title": "Police Assistance Cell",
        "welcome": "Welcome! I am the Thoothukudi District Police Assistance bot. How can I help you?",
        "placeholder": "Type your question here...",
        "buttons": ["Emergency contacts", "Police stations", "How to file a complaint?", "What is an FIR?", "About IPC sections"],
        "disclaimer": "This information is for general guidance only. It is not a substitute for legal advice. Please consult a legal professional for specific matters. This is an initiative by the Thoothukudi District Police."
    },
    "ta": {
        "title": "காவல்துறை உதவி செயலி",
        "welcome": "வணக்கம்! தூத்துக்குடி மாவட்ட காவல்துறை உதவி செயலிக்கு உங்களை வரவேற்கிறோம். நான் உங்களுக்கு எப்படி உதவ முடியும்?",
        "placeholder": "உங்கள் கேள்வியை இங்கு தட்டச்சு செய்யவும்...",
        "buttons": ["அவசர உதவி எண்கள்", "காவல் நிலையங்கள்", "புகார் அளிப்பது எப்படி?", "FIR என்றால் என்ன?", "IPC திருட்டு பற்றி"],
        "disclaimer": "இந்தத் தகவல்கள் பொதுவான வழிகாட்டுதலுக்காக மட்டுமே. இது சட்ட ஆலோசனைக்கு மாற்றாகாது. குறிப்பிட்ட வழக்குகளுக்கு சட்ட ஆலோசகரை அணுகவும். இது தூத்துக்குடி மாவட்ட காவல்துறையின் ஒரு முன்னோட்டச் செயலி."
    }
}

# --- Main App Logic ---

def main():
    st.set_page_config(page_title="CopBotChatbox", page_icon="🚨")

    st.sidebar.title("Language / மொழி")
    language = st.sidebar.radio("Choose Language", ('English', 'Tamil'), label_visibility="collapsed")
    lang_code = "ta" if language == "Tamil" else "en"

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- UPDATED INITIALIZATION LOGIC FOR FAST CLOUD DEPLOYMENT ---
    if "rag_chain" not in st.session_state:
        # The spinner will now appear for only a few seconds.
        with st.spinner("Bot is warming up..."):
            
            # 1. Load the pre-built vector database from disk.
            vector_db = load_vector_db()
            
            if vector_db:
                # 2. Create the effective RAG chain using the loaded vector database.
                st.session_state.rag_chain = create_effective_rag_chain(vector_db)
            else:
                st.error("Could not load the knowledge base. The application cannot start.")
                st.stop() # Stop the app if the knowledge base fails to load.

    # --- Main Chat Interface (This section remains unchanged) ---
    st.markdown(f"<h3 style='text-align: center;'>{UI_TEXT[lang_code]['title']}</h3>", unsafe_allow_html=True)

    # Display initial welcome message if chat is empty
    if not st.session_state.messages:
         st.session_state.messages.append({"role": "assistant", "content": UI_TEXT[lang_code]['welcome']})

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Display quick action buttons
    cols = st.columns(len(UI_TEXT[lang_code]['buttons']))
    for i, button_text in enumerate(UI_TEXT[lang_code]['buttons']):
        if cols[i].button(button_text):
            st.session_state.user_input_from_button = button_text
    
    # Handle user input (from chat input or button click)
    prompt = st.chat_input(UI_TEXT[lang_code]['placeholder'])
    if "user_input_from_button" in st.session_state and st.session_state.user_input_from_button:
        prompt = st.session_state.user_input_from_button
        st.session_state.user_input_from_button = None # Reset after use

    if prompt:
        # Add user message to chat history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            input_for_llm = translate_text(prompt, 'en') if lang_code == 'ta' else prompt
            
            response = st.session_state.rag_chain.invoke({"input": input_for_llm})
            response_text = response.get("answer", "Sorry, I encountered an issue.")
            
            final_response = translate_text(response_text, lang_code) if lang_code == 'ta' else response_text
        
        # Add assistant response to chat history and display it
        st.session_state.messages.append({"role": "assistant", "content": final_response})
        st.rerun()

    # Display disclaimer
    st.markdown(f"<p style='font-size: 0.8em; text-align: center; margin-top: 2em;'>{UI_TEXT[lang_code]['disclaimer']}</p>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()