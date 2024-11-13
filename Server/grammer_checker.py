import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load the grammar correction model
def load_grammar_model():
    """
    Loads the pre-trained grammar correction model and tokenizer from Hugging Face.

    This function loads the T5 model fine-tuned for grammatical error correction.
    It also detects whether a GPU is available and loads the model onto the appropriate device (CPU or GPU).

    Returns:
        tokenizer (T5Tokenizer): The tokenizer for the T5 model.
        model (T5ForConditionalGeneration): The pre-trained T5 model for grammar correction.
        torch_device (str): The device ('cuda' for GPU, 'cpu' for CPU) where the model is loaded.
    """
    model_name = 'abhinavsarkar/Google-T5-base-Grammatical_Error_Correction-Finetuned-C4-200M-550k'
    torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available, otherwise use CPU
    tokenizer = T5Tokenizer.from_pretrained(model_name)  # Load the T5 tokenizer
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(torch_device)  # Load the T5 model
    return tokenizer, model, torch_device


# Function for grammar correction
def correct_grammar(input_text, tokenizer, model, torch_device, num_return_sequences=2):
    """
    Corrects grammatical errors in the provided input text using the pre-trained grammar correction model.

    This function tokenizes the input text, passes it through the model to generate multiple possible corrections,
    and returns the corrected sentences.

    Args:
        input_text (str): The input sentence that needs grammar correction.
        tokenizer (T5Tokenizer): The tokenizer for the T5 model.
        model (T5ForConditionalGeneration): The pre-trained T5 model for grammar correction.
        torch_device (str): The device ('cuda' or 'cpu') where the model is loaded.
        num_return_sequences (int, optional): The number of different corrected sentence outputs to return. Default is 2.

    Returns:
        List[str]: A list of grammar-corrected sentences.
    """
    # Tokenize the input text and prepare it for the model
    batch = tokenizer([input_text], truncation=True, padding='max_length', max_length=64, return_tensors="pt").to(torch_device)
    
    # Generate corrected sentences using the model
    translated = model.generate(**batch, max_length=64, num_beams=4, num_return_sequences=num_return_sequences, temperature=1.5)
    
    # Decode the model's output and return the list of corrected sentences
    tgt_text = tokenizer.batch_decode(translated, skip_special_tokens=True)
    return tgt_text
