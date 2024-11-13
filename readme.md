# TextTweakAI: Your Personalized Spell & Grammar Checker

This is an intelligent **Spell and Grammar Checker** web app built using **Streamlit** and **Transformers**. It detects and corrects spelling and grammatical errors in your text. The app uses a pre-trained **T5 model** for grammar correction and a custom approach for spelling correction based on **Jaccard Similarity** and word frequency.

## Features
- **Spell Checker**: Detects spelling errors in your text and suggests corrections based on a corpus of literary works.
- **Grammar Checker**: Uses a fine-tuned **T5 model** to suggest grammar improvements.
- **Interactive UI**: Easily enter text, check for errors, and view suggestions side-by-side.
  
## Requirements
- `streamlit`
- `pandas`
- `textdistance`
- `torch`
- `transformers`
- `sentencepiece`
- `re`
- `collections`

Make sure to install the required dependencies by running:

```bash
pip install -r requirements.txt
```

## How to Use

1. **Enter Text**: In the app interface, enter the paragraph you want to check.
2. **Check Spelling**: Click **Check Spelling** to see detected spelling errors and suggestions for each incorrect word.
3. **Check Grammar**: Click **Check Grammar** to see grammatical corrections for each sentence in the input text.

### Spell Checker
- The spell checker compares each word in the input against a vocabulary loaded from multiple literary texts.
- It suggests corrections based on **Jaccard Similarity** (for measuring word similarity) and **word frequency** (for probability of usage).
  
### Grammar Checker
- The grammar checker uses a **T5 model** fine-tuned on grammatical error correction (GEC) to offer suggestions for improving sentence structure and grammar.
  
## How It Works
- **Spell Checking**: The spell checker uses **Jaccard Similarity** and word frequency probabilities to match and correct misspelled words in your text.
- **Grammar Checking**: The grammar checker generates multiple improved versions of each input sentence using the **T5 model** fine-tuned for grammatical error correction.

## Model Details

### Grammar Checker Model
The Grammar Checker is based on a fine-tuned **T5** model designed for grammatical error correction. This model has been trained on a large dataset of text and is fine-tuned for high accuracy in generating grammatically correct sentences.

- 🔗 [Model on Hugging Face](https://huggingface.co/abhinavsarkar/Google-T5-base-Grammatical_Error_Correction-Finetuned-C4-200M-550k)
- 📊 [Dataset on Hugging Face](https://huggingface.co/datasets/abhinavsarkar/C4-200m-550k-Determiner)
- 📂 [Original Dataset URL](https://www.kaggle.com/datasets/felixstahlberg/the-c4-200m-dataset-for-gec)
- 🛠️ [GitHub Repository](https://github.com/AbhinavSarkarr/Spell-and-Grammer-Checker)

You can also check out the Synthetic DataGenerator used for creating the 200M data C4 original datset. [here](https://github.com/google-research-datasets/C4_200M-synthetic-dataset-for-grammatical-error-correction).

### Spell Checker Information
The Spell Checker uses a vocabulary from multiple literary texts and publicly available resources. It uses **Jaccard Similarity** and **word frequency probabilities** to detect spelling errors and suggest the closest correct words.

- 📂 [Corpus Resource](https://drive.google.com/drive/u/0/folders/1WsvpWHKUv3OI2mRce-NPg4HsVPyhfk0e)

## Running the App

To run the app locally, simply execute the following command in the terminal:

```bash
streamlit run app.py
```

This will launch the Streamlit app in your browser, where you can interact with the Spell and Grammar Checker.

## Deployment

This app is deployed on **Hugging Face Spaces**. You can access the live version of the app [here](https://huggingface.co/spaces/abhinavsarkar/TextTweakAI).

## Contributing

Feel free to fork this repository, submit issues, or create pull requests to improve the app.
