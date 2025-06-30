from deep_translator import GoogleTranslator

# Initialize the translator once


def translate_text(text: str, target_lang: str = "ta") -> str:
    """
    Translates text to the destination language.
    Args:
        text (str): The text to translate.
        dest_language (str): The destination language code (e.g., 'en', 'ta').
    Returns:
        str: The translated text.
    """
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception as e:
        return f"Translation error: {str(e)}"