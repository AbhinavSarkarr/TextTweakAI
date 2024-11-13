import streamlit as st
import pandas as pd
import textdistance
import re
from collections import Counter
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Set the page configuration as the first Streamlit command
st.set_page_config(page_title="Spell & Grammar Checker", layout="wide")

# Load the grammar correction model
@st.cache_resource
def load_grammar_model():
    model_name = 'abhinavsarkar/Google-T5-base-Grammatical_Error_Correction-Finetuned-C4-200M-550k'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(torch_device)
    return tokenizer, model, torch_device

tokenizer, model, torch_device = load_grammar_model()

# Load vocabulary for spell checking (optimized loading)
@st.cache_resource
def load_vocabulary():
    file_paths = ['Vocabulary/book.txt', 'Vocabulary/alice_in_wonderland.txt', 'Vocabulary/big.txt', 'Vocabulary/shakespeare.txt']
    words = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            file_name_data = f.read().lower()
            words += re.findall(r'\w+', file_name_data)
    V = set(words)
    word_freq = Counter(words)
    probs = {k: word_freq[k] / sum(word_freq.values()) for k in word_freq}
    return V, word_freq, probs

V, word_freq, probs = load_vocabulary()

# Precompute Jaccard similarity scores for spell check
def precompute_similarities(input_word):
    input_word = input_word.lower()
    sim = [1 - (textdistance.Jaccard(qval=2).distance(v, input_word)) for v in word_freq.keys()]
    return sim

def my_autocorrect(input_paragraph, top_n=5):
    input_paragraph = input_paragraph.lower()
    words_in_paragraph = re.findall(r'\w+', input_paragraph)
    incorrect_words = []
    corrected_words = []
    for word in words_in_paragraph:
        if word not in V:
            sim = precompute_similarities(word)
            df = pd.DataFrame.from_dict(probs, orient='index').reset_index()
            df = df.rename(columns={'index': 'Word', 0: 'Prob'})
            df['Similarity'] = sim
            output = df.sort_values(['Similarity', 'Prob'], ascending=False).head(top_n)
            output = output[['Word', 'Similarity', 'Prob']].reset_index(drop=True)
            output.index = output.index + 1
            incorrect_words.append(word)
            corrected_words.append(output)
    return incorrect_words, corrected_words

# Function for grammar correction
def correct_grammar(input_text, num_return_sequences=2):
    batch = tokenizer([input_text], truncation=True, padding='max_length', max_length=64, return_tensors="pt").to(torch_device)
    translated = model.generate(**batch, max_length=64, num_beams=4, num_return_sequences=num_return_sequences, temperature=1.5)
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text

# Streamlit app layout
def main():
    st.title("📚 Intelligent Spell & Grammar Checker")
    st.markdown("""
        Welcome to the **Spell & Grammar Checker**! This app is designed to help you improve your writing by detecting and correcting spelling and grammar errors. Simply enter a paragraph below and let the app do the rest. Each section provides unique suggestions to refine your text.
    """)

    paragraph = st.text_area("✨ Enter a paragraph to check for spelling and grammar issues:", height=200)

    # Two side-by-side sections
    col1, col2 = st.columns(2)

    # Initialize session state for storing results
    if 'spelling_results' not in st.session_state:
        st.session_state.spelling_results = None
    if 'grammar_results' not in st.session_state:
        st.session_state.grammar_results = None

    with col1:
        st.header("🔍 Spell Checker")
        st.markdown("""
            **About the Spell Checker:**  
            Our spell checker uses a vocabulary from multiple literary texts to detect potential misspellings. It offers suggestions ranked by similarity and probability, helping you to identify and correct errors with ease.  
            **How to use:**  
            Enter a paragraph and click **Check Spelling** to see any misspelled words along with suggestions.
        """)
        
        if st.button("Check Spelling"):
            if paragraph:
                with st.spinner("Checking spelling..."):
                    incorrect_words, corrected_words = my_autocorrect(paragraph)
                    if incorrect_words:
                        st.session_state.spelling_results = (incorrect_words, corrected_words)
                    else:
                        st.session_state.spelling_results = ("✅ No spelling errors detected!", [])
            else:
                st.warning("Please enter a paragraph to check for spelling.")

        if st.session_state.spelling_results:
            incorrect_words, corrected_words = st.session_state.spelling_results
            if isinstance(incorrect_words, str):
                st.success(incorrect_words)
            else:
                st.subheader("🔴 Spelling Errors & Suggestions:")
                for i, word in enumerate(incorrect_words):
                    st.write(f"**Misspelled Word**: `{word}`")
                    with st.expander(f"Suggestions for `{word}`"):
                        suggestions_df = corrected_words[i]
                        st.table(suggestions_df[['Word', 'Similarity', 'Prob']])

    with col2:
        st.header("📝 Grammar Checker")
        st.markdown("""
            **About the Grammar Checker:**  
            Powered by a fine-tuned T5 model, our grammar checker analyzes each sentence for potential errors in structure, tense, and word choice. It offers refined suggestions to enhance readability and grammatical accuracy.  
            **How to use:**  
            Enter a paragraph and click **Check Grammar** to review each sentence with suggested improvements.
        """)
        
        if st.button("Check Grammar"):
            if paragraph:
                with st.spinner("Checking grammar..."):
                    sentences = re.split(r'(?<=[.!?]) +', paragraph)
                    grammar_results = []
                    for sentence in sentences:
                        if sentence.strip():
                            corrected_sentences = correct_grammar(sentence, num_return_sequences=2)
                            grammar_results.append((sentence, corrected_sentences))
                    st.session_state.grammar_results = grammar_results
            else:
                st.warning("Please enter a paragraph to check for grammar.")

        if st.session_state.grammar_results:
            st.subheader("🔵 Grammar Corrections:")
            for sentence, corrected_sentences in st.session_state.grammar_results:
                with st.expander(f"**Original Sentence:** {sentence}", expanded=True):
                    st.write("### Suggestions:")
                    for corrected_sentence in corrected_sentences:
                        st.write(f"- {corrected_sentence}")

    # Model details section
    st.markdown("---")
    st.header("📘 Grammar Checker Information")

    st.markdown("""
    ### Grammar Checker Model  
    The Grammar Checker model, fine-tuned for grammatical error correction (GEC), is ideal for enhancing writing quality across various domains. Below, you'll find relevant resources related to this model's development and usage.

    - 🔗 **[Finetuned Model on Hugging Face](https://huggingface.co/abhinavsarkar/Google-T5-base-Grammatical_Error_Correction-Finetuned-C4-200M-550k)**  
    Access the model details, fine-tuning specifics, and download options on Hugging Face.

    - 📊 **[Used Dataset on Hugging Face](https://huggingface.co/datasets/abhinavsarkar/C4-200m-550k-Determiner)**  
    Explore the pre-processed dataset used to train this model.

    - 📂 **[Original Dataset URL](https://www.kaggle.com/datasets/felixstahlberg/the-c4-200m-dataset-for-gec)**  
    This dataset contains 200 million sentences with diverse structures, hosted on Kaggle.

    - 🛠️ **[GitHub Repository](https://github.com/AbhinavSarkarr/Spell-and-Grammer-Checker)**  
    Access the code repository for dataset preparation, model training, and additional development resources.
    """)

    # Spell Checker Information
    st.markdown("---")
    st.header("🔍 Spell Checker Information")

    st.markdown("""
    ### Spell Checker  
    The Spell Checker leverages a corpus containing multiple text resources to suggest corrections for spelling errors. The algorithm uses **Jaccard Similarity** and **Relative Probability** to identify the closest matches to the input words, ensuring accuracy in suggestions.

    - 📂 **[Corpus Resource](https://drive.google.com/drive/u/0/folders/1WsvpWHKUv3OI2mRce-NPg4HsVPyhfk0e)**  
    The vocabulary for this checker is based on a collection of literary works and publicly available texts.
    """)

# Run the app
if __name__ == "__main__":
    main()
