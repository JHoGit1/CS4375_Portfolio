# Jonathan Ho
# CS 4395

'''
This script takes in a text file and first calculates the lexical diversity. Then it preprocesses the
text file and is tokenized and lemmatized to get a list of all tokens and a list of nouns. The lists are used
to make a list of words to be used for a guessing game.
'''

import sys
import nltk
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from pathlib import Path
from random import randint

# Private function to choose a word from a given list for the guessing game
def _select_word(guess_list):
    unknown_w = ""

    # Select a random word from the word list for the game
    gw_key = guess_list[randint(0, 49)]

    # Create the string to output the status of the guessed word
    for y in range(len(gw_key)):
        unknown_w = unknown_w + "_" + " "

    return gw_key, unknown_w

# Function that plays a guessing game with a given list of words
def guessing_game(word_list):
    score = 5

    # Select a word and get the guess word key and the string to update the status on guesses
    guess_word_key, unknown_word = _select_word(word_list)

    print("\nLet's play a word guessing game!")

    # Loop for the guessing game
    while score > -1:
        print(unknown_word)
        guess = input("Guess a letter: ")

        # Terminate early if guess is !
        if guess == "!":
            print("! detected, game terminated")
            print("Total score:", score)
            break
        # Lose a point for bad guess
        elif guess not in guess_word_key:
            score = score - 1
            print("Sorry, guess again. Score is", score)
        # Gain a point for correct guess and update word status
        else:
            score = score + 1
            print("Right! Score is", score)

            # Loop to fill underscores with guess letter
            for g in range(len(guess_word_key)):
                if guess == guess_word_key[g]:
                    unknown_word = unknown_word[:g * 2] + guess + unknown_word[g * 2 + 1:]

        # If word is solved, print current results and generate new word
        if "_" not in unknown_word:
            print(unknown_word)
            print("You solved it!")
            print("\nCurrent score:", score)
            guess_word_key, unknown_word = _select_word(word_list)
            print("\nGuess another word")

    # Post game results
    if guess != "!":
        print("You did not guess the word. The word was", guess_word_key)

# Function to preprocess the raw text
def preprocess_text(text):
    # Tokenize the raw text
    text_tok = nltk.word_tokenize(text)

    # Lowercase tokenized text
    edit_text = [t.lower() for t in text_tok]

    # Remove punctuation and stop words from tokenized text
    edit_text = [t for t in edit_text if t.isalpha() and t not in stopwords.words('english')]

    # Remove tokens of length 5 or less
    # Must be > 6 because I think it is including \n hidden in the text as a character
    edit_text = [t for t in edit_text if len(t) > 6]

    # Lemmatize the tokens and make unique
    wnl = WordNetLemmatizer()
    lemmas = [wnl.lemmatize(t) for t in edit_text]
    unique_lemmas = list(set(lemmas))

    # Part-of-speech tag on the unique lemmas
    lemma_tags = nltk.pos_tag(unique_lemmas)

    # Print the first 20 tagged words
    for w in range(20):
        print(lemma_tags[w])

    # Create a list of only the nouns
    dict_nouns = [t for t, pos in lemma_tags if (pos == 'NN' or pos == 'NNS' or pos == 'NNP' or pos == 'NNPS')]

    # Print number of tokens and nouns
    print("\nNumber of tokens post-processing:", len(edit_text))
    print("Number of nouns:", len(dict_nouns))

    # Return processed tokens and list of nouns
    return edit_text, dict_nouns

if __name__ == '__main__':
    # Read in the raw text into a variable and tokenize
    raw_text = Path(sys.argv[1]).read_text()
    text_token = nltk.word_tokenize(raw_text)

    # Get all the unique tokens in the text and calculate lexical diversity (unique over total)
    uni_tok = set(text_token)
    print("\nLexical diversity: %.2f" % (len(uni_tok) / len(text_token)))

    # Preprocess raw text and return list of processed tokens and all nouns
    processed_tok, noun_list = preprocess_text(raw_text)

    # Make a dictionary of nouns and their amount of appearance in the token list
    dict_noun_count = {noun_list: processed_tok.count(noun_list) for noun_list in processed_tok}

    # Print out the top 50 most common nouns in descending order and add nouns into a list
    sorted_dict_nouns = Counter(dict_noun_count)
    list_nouns = []

    print("\nTop 50 most common nouns [noun:count]: ")
    count = 1

    for n, c in sorted_dict_nouns.most_common(50):
        print(str(count) + ". ", end="")
        print("%s:%d" % (n, c))
        count = count + 1
        list_nouns.append(n)

    # Guessing game function
    guessing_game(list_nouns)