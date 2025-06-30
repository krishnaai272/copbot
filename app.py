import streamlit as st
from dotenv import load_dotenv
import os

from src.rag_processor import create_knowledge_base_from_pdfs
from src.llm_handler import create_rag_chain_from_retriever
from src.utils import translate_text

load_dotenv()

# --- UI Configuration (remains unchanged) ---
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

    if "rag_chain" not in st.session_state:
        with st.spinner("Bot is warming up... This may take a moment."):
            pdf_files = [
                os.path.join("data", "Case section Tamilnadu.pdf"),
                os.path.join("data", "lawrules.pdf"),
                os.path.join("data", "lawruleone.pdf"),
                os.path.join("data", "lawrulestwo.pdf")
            ]
            
            # This function now returns a fully configured retriever
            retriever = create_knowledge_base_from_pdfs(pdf_files)
            
            if retriever:
                # We pass the retriever directly to the chain creation function
                st.session_state.rag_chain = create_rag_chain_from_retriever(retriever)
            else:
                st.stop()

    # --- The rest of the app logic remains the same ---
    st.markdown(f"<h3 style='text-align: center;'>{UI_TEXT[lang_code]['title']}</h3>", unsafe_allow_html=True)

    if not st.session_state.messages:
         st.session_state.messages.append({"role": "assistant", "content": UI_TEXT[lang_code]['welcome']})

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    cols = st.columns(len(UI_TEXT[lang_code]['buttons']))
    for i, button_text in enumerate(UI_TEXT[lang_code]['buttons']):
        if cols[i].button(button_text):
            st.session_state.user_input_from_button = button_text
    
    prompt = st.chat_input(UI_TEXT[lang_code]['placeholder'])
    if "user_input_from_button" in st.session_state and st.session_state.user_input_from_button:
        prompt = st.session_state.user_input_from_button
        st.session_state.user_input_from_button = None

    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.spinner("Thinking..."):
            input_for_llm = translate_text(prompt, 'en') if lang_code == 'ta' else prompt
            
            response = st.session_state.rag_chain.invoke({"input": input_for_llm})
            response_text = response.get("answer", "Sorry, I encountered an issue.")
            
            final_response = translate_text(response_text, lang_code) if lang_code == 'ta' else response_text
        
        st.session_state.messages.append({"role": "assistant", "content": final_response})
        st.rerun()

    st.markdown(f"<p style='font-size: 0.8em; text-align: center; margin-top: 2em;'>{UI_TEXT[lang_code]['disclaimer']}</p>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()