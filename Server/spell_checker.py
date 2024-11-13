import pandas as pd
import re
from collections import Counter
import textdistance

# Load vocabulary for spell checking (optimized loading)
def load_vocabulary():
    """
    Loads and processes vocabulary from multiple text files for spell checking.

    This function reads text from a set of files, extracts words, and creates a vocabulary.
    It also calculates the frequency of each word and the probability distribution based on word frequency.

    Returns:
        V (set): A set containing all unique words from the text files.
        word_freq (Counter): A counter object with word frequencies from the vocabulary.
        probs (dict): A dictionary with the probability of each word based on its frequency.
    """
    file_paths = ['Vocabulary/book.txt', 'Vocabulary/alice_in_wonderland.txt', 'Vocabulary/big.txt', 'Vocabulary/shakespeare.txt']  #you can get these files from the corpus resouces given in the readme file 
    words = []
    for file_path in file_paths:
        with open(file_path, 'r') as f:
            file_name_data = f.read().lower()
            words += re.findall(r'\w+', file_name_data)
    
    V = set(words)  # Unique words
    word_freq = Counter(words)  # Word frequency count
    probs = {k: word_freq[k] / sum(word_freq.values()) for k in word_freq}  # Word probability distribution
    return V, word_freq, probs


# Precompute Jaccard similarity scores for spell check
def precompute_similarities(input_word, word_freq):
    """
    Computes the Jaccard similarity between the input word and all words in the vocabulary.

    This function calculates the similarity between the input word and each word in the vocabulary
    using the Jaccard similarity metric. The lower the distance, the more similar the words are.

    Args:
        input_word (str): The word for which the similarities are to be calculated.
        word_freq (Counter): A counter object containing word frequencies in the vocabulary.

    Returns:
        list: A list of similarity scores between the input word and each word in the vocabulary.
    """
    input_word = input_word.lower()  # Ensure case insensitivity
    sim = [1 - (textdistance.Jaccard(qval=2).distance(v, input_word)) for v in word_freq.keys()]
    return sim


def my_autocorrect(input_paragraph, V, word_freq, probs, top_n=5):
    """
    Performs spell checking and correction on an input paragraph.

    This function breaks the input paragraph into words, identifies words not in the vocabulary,
    and suggests corrections for them based on similarity and word probability. It returns the 
    misspelled words along with their top suggested corrections.

    Args:
        input_paragraph (str): The input text to be checked for spelling errors.
        V (set): A set containing all unique words in the vocabulary.
        word_freq (Counter): A counter object containing word frequencies.
        probs (dict): A dictionary with the probability distribution of words in the vocabulary.
        top_n (int, optional): The number of suggestions to return for each misspelled word. Default is 5.

    Returns:
        incorrect_words (list): A list of words that are considered misspelled.
        corrected_words (list): A list of dataframes containing the top suggested corrections for each misspelled word.
    """
    input_paragraph = input_paragraph.lower()  # Convert the input to lowercase for case-insensitive matching
    words_in_paragraph = re.findall(r'\w+', input_paragraph)  # Tokenize the paragraph into words
    incorrect_words = []  # List to store incorrect words
    corrected_words = []  # List to store the suggestions for corrections
    
    for word in words_in_paragraph:
        if word not in V:  # If the word is not in the vocabulary
            sim = precompute_similarities(word, word_freq)  # Calculate similarity scores
            df = pd.DataFrame.from_dict(probs, orient='index').reset_index()  # Create a DataFrame from the word probabilities
            df = df.rename(columns={'index': 'Word', 0: 'Prob'})  # Rename columns for clarity
            df['Similarity'] = sim  # Add similarity scores to the DataFrame
            output = df.sort_values(['Similarity', 'Prob'], ascending=False).head(top_n)  # Sort and get top N suggestions
            output = output[['Word', 'Similarity', 'Prob']].reset_index(drop=True)  # Select relevant columns
            output.index = output.index + 1  # Adjust index to start from 1 instead of 0
            incorrect_words.append(word)  # Append the incorrect word to the list
            corrected_words.append(output)  # Append the suggestions for the word
    
    return incorrect_words, corrected_words
