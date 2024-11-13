Let's walk through these steps with an example to understand how an autocorrect model works.

### Example:
Suppose we have the misspelled word `"drea"` and want to correct it to a real word. 

### Step 1: Identify Misspelled Word
1. **Look up the word** `"drea"` in a dictionary of valid words.
2. Since `"drea"` is not found in the dictionary, we flag it as a misspelled word and move to the next step to generate possible corrections.

### Step 2: Find 'n' Strings Edit Distance Away
1. **Edit Distance**: To find possible corrections, we’ll generate all words that are a small number of edits (say, `n = 1 or 2`) away from `"drea"`. 
2. **Types of Edits**:
   - **Insert**: Add a letter at any position in `"drea"` to create new words, such as `"dream"`, `"dread"`, or `"bread"`.
   - **Delete**: Remove a letter, giving options like `"rea"`, `"dea"`, or `"dra"`.
   - **Switch**: Swap adjacent letters, creating `"dera"`, `"drea"`, etc.
   - **Replace**: Replace one letter with another, yielding words like `"area"`, `"drop"`, or `"tree"`.
   
3. This process produces a **candidate list** of words that are one or two edits away from `"drea"`.

   **Candidate List Example**: `["dream", "dread", "area", "dear", "drop", "tree", "bread", "dare"]`

### Step 3: Filtering of Candidates
1. Now, we check each word in the candidate list to see if it is a real word by looking it up in our dictionary.
2. Words not found in the dictionary are removed. This ensures we’re only working with real, correctly spelled words.

   **Filtered Candidate List Example**: `["dream", "dread", "area", "dear", "tree", "bread", "dare"]`

### Step 4: Calculate Probabilities of Words
1. **Word Frequencies**: To choose the best suggestion, we look at the frequency of each word in a large corpus (a collection of text data). The more frequently a word appears, the more likely it is the correct choice.
2. **Calculate Probability**: For each word in the filtered candidate list, calculate the probability based on its frequency in the corpus.
3. **Select Most Likely Word**: The word with the highest probability becomes the model’s suggestion for correction.

   **Example Probabilities**:
   - "dream" — 0.4
   - "dread" — 0.1
   - "area" — 0.05
   - "dear" — 0.2
   - "tree" — 0.05
   - "bread" — 0.15
   - "dare" — 0.05

   Here, **"dream"** has the highest probability (0.4), so the model would suggest `"dream"` as the corrected word for `"drea"`.

### Final Output:
Based on the autocorrect model, the best suggestion for `"drea"` is `"dream"`.

This process allows the autocorrect model to intelligently suggest corrections by balancing closeness in spelling with likelihood based on word usage.



For creating the dictionary we have used the book: 
1. The Project Gutenberg EBook of Moby Dick; or The Whale, by Herman Melville, Link: https://github.com/amankharwal/Website-data/blob/master/book.txt
2. Alice_in_wonderland, Link: https://github.com/Wanghley/Word-Autocorrection-NLP/blob/main/alice_in_wonderland.txt