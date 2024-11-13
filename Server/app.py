import streamlit as st
from spell_checker import my_autocorrect, load_vocabulary
from grammer_checker import correct_grammar, load_grammar_model
import re

# Set the page configuration as the first Streamlit command
st.set_page_config(page_title="Spell & Grammar Checker", layout="wide")

# Load models and vocabulary
tokenizer, model, torch_device = load_grammar_model()
V, word_freq, probs = load_vocabulary()

def main():
    """
    Main function to launch the Streamlit app. This function sets up the layout, handles user input,
    and interacts with the spell and grammar checking components.
    """
    # Set title and description for the app
    st.title("📚 Intelligent Spell & Grammar Checker")
    st.markdown("""
        Welcome to the **Spell & Grammar Checker**! This app is designed to help you improve your writing by detecting and correcting spelling and grammar errors. 
        Simply enter a paragraph below and let the app do the rest. Each section provides unique suggestions to refine your text.
    """)

    # Text area for user to input paragraph to check
    paragraph = st.text_area("✨ Enter a paragraph to check for spelling and grammar issues:", height=200)

    # Two side-by-side columns for displaying results
    col1, col2 = st.columns(2)

    # Initialize session state to store spelling and grammar results
    if 'spelling_results' not in st.session_state:
        st.session_state.spelling_results = None
    if 'grammar_results' not in st.session_state:
        st.session_state.grammar_results = None

    with col1:
        # Spell Checker Section
        st.header("🔍 Spell Checker")
        
        # Button to trigger spelling check
        if st.button("Check Spelling"):
            if paragraph:
                with st.spinner("Checking spelling..."):
                    # Run spell check on the paragraph
                    incorrect_words, corrected_words = my_autocorrect(paragraph, V, word_freq, probs)
                    
                    # Store results in session state
                    if incorrect_words:
                        st.session_state.spelling_results = (incorrect_words, corrected_words)
                    else:
                        st.session_state.spelling_results = ("✅ No spelling errors detected!", [])
            else:
                st.warning("Please enter a paragraph to check for spelling.")

        # Display spelling check results
        if st.session_state.spelling_results:
            incorrect_words, corrected_words = st.session_state.spelling_results
            if isinstance(incorrect_words, str):
                st.success(incorrect_words)  # No errors detected
            else:
                st.subheader("🔴 Spelling Errors & Suggestions:")
                for i, word in enumerate(incorrect_words):
                    st.write(f"**Misspelled Word**: `{word}`")
                    with st.expander(f"Suggestions for `{word}`"):
                        # Show correction suggestions for each word
                        suggestions_df = corrected_words[i]
                        st.table(suggestions_df[['Word', 'Similarity', 'Prob']])

    with col2:
        # Grammar Checker Section
        st.header("📝 Grammar Checker")
        
        # Button to trigger grammar check
        if st.button("Check Grammar"):
            if paragraph:
                with st.spinner("Checking grammar..."):
                    # Split paragraph into sentences
                    sentences = re.split(r'(?<=[.!?]) +', paragraph)
                    grammar_results = []
                    
                    # Check grammar for each sentence
                    for sentence in sentences:
                        if sentence.strip():  # Avoid checking empty sentences
                            corrected_sentences = correct_grammar(sentence, tokenizer, model, torch_device)
                            grammar_results.append((sentence, corrected_sentences))
                    
                    # Store grammar results in session state
                    st.session_state.grammar_results = grammar_results
            else:
                st.warning("Please enter a paragraph to check for grammar.")

        # Display grammar check results
        if st.session_state.grammar_results:
            st.subheader("🔵 Grammar Corrections:")
            for sentence, corrected_sentences in st.session_state.grammar_results:
                with st.expander(f"**Original Sentence:** {sentence}", expanded=True):
                    st.write("### Suggestions:")
                    for corrected_sentence in corrected_sentences:
                        st.write(f"- {corrected_sentence}")

# Run the app
if __name__ == "__main__":
    main()
