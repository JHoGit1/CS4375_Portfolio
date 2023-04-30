# Jonathan Ho
# CS 4395

'''This program takes in pickled dictionaries to the language of lines in a test file.
Then it compares the results with a solution file'''

import pickle
from nltk import word_tokenize
from nltk.util import ngrams

# Calculate correct accuracy between two files
def correct(test, ans):
    num_correct = 0
    total = 0

    for test_line, ans_line in zip(test.readlines(), ans.readlines()):
        total += 1
        if test_line.rstrip() == ans_line.rstrip():
            num_correct += 1

    print("Correct Accuracy: " + "%.2f" % ((num_correct / total) * 100) + "%")

# Calculate incorrect accuracy between the two files and output wrong lines
def incorrect(test, ans):
    num_incorrect = 0
    total = 0
    incorrect_list = []

    for test_line, ans_line in zip(test.readlines(), ans.readlines()):
        total += 1
        if test_line.rstrip() != ans_line.rstrip():
            num_incorrect += 1
            incorrect_list.append(total)

    print("Incorrect Accuracy: " + "%.2f" % ((num_incorrect / total) * 100) + "%")
    print("Incorrect lines:")
    print(incorrect_list)

# Private function to calculate the probability given a language's dictionaries and a line
def _calc_prob(text, u_dict, b_dict, v):
    unigrams_split = word_tokenize(text)
    bigrams_split = list(ngrams(unigrams_split, 2))
    line_prob = 1

    for bigram in bigrams_split:
        n = b_dict[bigram] if bigram in b_dict else 0
        d = u_dict[bigram[0]] if bigram[0] in u_dict else 0

        line_prob *= (n + 1) / (d + v)

    return line_prob

# Calculate which language the line is
def lang_prob(en_u, en_b, fr_u, fr_b, ita_u, ita_b):
    prob_file = open(r'data\test_prob.txt', 'w')
    test_file = open(r'data\LangId.test', 'r', encoding="utf-8")
    test_line = test_file.readline()
    v = len(en_u) + len(fr_u) + len(ita_u)
    line_num = 1

    # Calculate which language the line is and write to a file
    while test_line:
        eng_prob = _calc_prob(test_line, en_u, en_b, v)
        fr_prob = _calc_prob(test_line, fr_u, fr_b, v)
        ita_prob = _calc_prob(test_line, ita_u, ita_b, v)

        if eng_prob > fr_prob and eng_prob > ita_prob:
            prob_file.write(str(line_num) + " English\n")
        elif fr_prob > eng_prob and fr_prob > ita_prob:
            prob_file.write(str(line_num) + " French\n")
        else:
            prob_file.write(str(line_num) + " Italian\n")

        line_num += 1
        test_line = test_file.readline()

    prob_file.close()
    test_file.close()

if __name__ == '__main__':
    # Read in the unigram and bigram dictionaries made
    with open(r'data\eng_unigram_dict.p', 'rb') as a:
        eng_unigram_dict = pickle.load(a)
    with open(r'data\eng_bigram_dict.p', 'rb') as b:
        eng_bigram_dict = pickle.load(b)
    with open(r'data\fr_unigram_dict.p', 'rb') as c:
        fr_unigram_dict = pickle.load(c)
    with open(r'data\fr_bigram_dict.p', 'rb') as d:
        fr_bigram_dict = pickle.load(d)
    with open(r'data\ita_unigram_dict.p', 'rb') as e:
        ita_unigram_dict = pickle.load(e)
    with open(r'data\ita_bigram_dict.p', 'rb') as f:
        ita_bigram_dict = pickle.load(f)

    # Calculate probabilities of each language and output to a file
    lang_prob(eng_unigram_dict, eng_bigram_dict, fr_unigram_dict, fr_bigram_dict, ita_unigram_dict, ita_bigram_dict)

    # Compare the test file with the solution file and print out the correct and incorrect accuracy
    with open(r'data\test_prob.txt', 'r') as test_cor, open(r'data\LangId.sol', 'r') as sol_cor:
        correct(test_cor, sol_cor)

    with open(r'data\test_prob.txt', 'r') as test_inc, open(r'data\LangId.sol', 'r') as sol_inc:
        incorrect(test_inc, sol_inc)