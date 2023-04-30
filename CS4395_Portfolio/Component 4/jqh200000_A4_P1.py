# Jonathan Ho
# CS 4395

'''This program takes in a text file from a data folder and creats a unigram and bigram dictionary using ngram'''

from nltk import word_tokenize
from nltk.util import ngrams
import pickle

# Private function to create a unigram dictionary
def _unigram_dict(unigram_list, txt_file):
    unigram_dict = dict()

    for unigram in unigram_list:
        unigram_dict[unigram] = txt_file.count(unigram[0])

    return unigram_dict

# Private function to create a bigram dictionary
def _bigram_dict(bigram_list, txt_file):
    bigram_dict = dict()

    for bigram in bigram_list:
        bigram_dict[bigram] = txt_file.count(bigram[0] + ' ' + bigram[1])

    return bigram_dict

# Function to take in file, remove newlines, and tokenize text, then create dictionaries of the unigrams and bigrams
def text_to_dict(filepath):
    # Open file with encoding specified to 'utf-8' for any language
    lang_file = open(filepath, 'r', encoding="utf-8")
    read_file = lang_file.readlines()

    # Remove newlines
    edit_file = [s.rstrip('\n') for s in read_file]

    # Convert list back to string
    noline_string = ' '.join(map(str, edit_file))

    # Tokenize text
    token = word_tokenize(noline_string)

    # Make unigram and bigram generator objects
    unigrams = ngrams(token, 1)
    bigrams = ngrams(token, 2)

    # Create dictionaries from the generator objects
    unigram_dict = _unigram_dict(unigrams, noline_string)
    bigram_dict = _bigram_dict(bigrams, noline_string)

    lang_file.close()
    return unigram_dict, bigram_dict

if __name__ == '__main__':
    # Create unigram and bigram dictionaries for the english, french, and italian texts
    eng_unigram_dict, eng_bigram_dict = text_to_dict(r'data\LangId.train.English')
    fr_unigram_dict, fr_bigram_dict = text_to_dict(r'data\LangId.train.French')
    ita_unigram_dict, ita_bigram_dict = text_to_dict(r'data\LangId.train.Italian')

    # Pickle the dictionaries
    pickle.dump(eng_unigram_dict, open(r'data\eng_unigram_dict.p', 'wb'))
    pickle.dump(eng_bigram_dict, open(r'data\eng_bigram_dict.p', 'wb'))
    pickle.dump(fr_unigram_dict, open(r'data\\fr_unigram_dict.p', 'wb'))
    pickle.dump(fr_bigram_dict, open(r'data\\fr_bigram_dict.p', 'wb'))
    pickle.dump(ita_unigram_dict, open(r'data\ita_unigram_dict.p', 'wb'))
    pickle.dump(ita_bigram_dict, open(r'data\ita_bigram_dict.p', 'wb'))