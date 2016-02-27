'''
    This is a module to calculate prior probabilities and of the classes
    'sponsored' and 'unsponsored' and conditional probabilities all the prominent
    words identified from every documents.

    Prior Probabilities - P(C) = N(C)/N
    Conditional Probabilities - P(w|C) = (count(w,C)+1)/(count(C)+|V|)
'''
import sampler
import operator, numpy

word_frequencies = {}
vocab_count = {}
all_words_count = {}
prior_probabilities = {}

def init_params(of_class, word_counts):
    global word_frequencies, vocab_count, all_words_count
    word_frequencies[of_class] = word_counts
    vocab_count[of_class] = sum(1 for key in numpy.unique(word_counts.keys()))
    all_words_count[of_class] = sum(value for value in word_counts.values())

def calculate_prior_probabilities(training_file, classes):
    count_of = {}
    total_rows = len(training_file)
    for each_class, y_value in classes.items():
        count_of[each_class] = sampler.count_of_class(training_file, y_value)
        prior_probabilities[each_class] = count_of[each_class] / total_rows


def get_word_freq(word, of_class):
    return word_frequencies.get(of_class).get(word, 0)

'''
   P(word|Class) = (count(w,C)+1)/(count(C)+|V|)
   Note: Vocab_count should have counts for all classes as |V| is the vocab in all classes.
'''
def likelihood_of_word(word, given_class):
    return (get_word_freq(word,given_class)+1) / (all_words_count.get(given_class) + sum(count for count in vocab_count.values()))

def calculate_conditional_probabilities(words, class_names):
    word_probabilities = {}
    for each_class in class_names:
        word_probabilities[each_class] = {}
        for each_word in set(words):
            word_probabilities[each_class][each_word] = likelihood_of_word(each_word, each_class)
    return word_probabilities

def nb_classifier(test_blob, classes):
    word_probabilities = calculate_conditional_probabilities(test_blob.words, classes.keys())
    class_probabilities = {}
    for each_class in classes.keys():
        class_probabilities[each_class] = prior_probabilities[each_class]
        for each_word in set(test_blob.words):
            class_probabilities[each_class] *= pow(word_probabilities[each_class][each_word], test_blob.words.count(each_word))

    return max(class_probabilities.items(), key=operator.itemgetter(1))[0]

def main():
    print('Naive Bayes')

if __name__ == '__main__':
    main()