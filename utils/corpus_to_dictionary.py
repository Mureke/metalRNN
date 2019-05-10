import io
from settings.data_norm_constants import MIN_WORD_FREQUENCY, SEQUENCE_LEN
import keras
import numpy as np


def corpus_to_dictionary(path: str):
    with io.open(path) as f:
        # Get words from corpus file
        text = f.read().lower().replace('\n', ' \n ').replace('\\', ' \n ')
        text_in_words = keras.preprocessing.text.text_to_word_sequence(text, filters='"#$%&()*+,-./:;<=>@[\\]^_`{|}~\t', lower=True,
                                                      split=' ')
        print('Corpus length in words:', len(text_in_words))
        if ('\n' in text_in_words):
            index = text_in_words.index('\n')
            test = text_in_words[index]

        # Count how many times word appears in text_in_words
        word_freq = {}
        for word in text_in_words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Get ignored words and add them to ignored_words_set
        ignored = set()
        for k, v in word_freq.items():
            if word_freq[k] < MIN_WORD_FREQUENCY:
                ignored.add(k)

        words = set(text_in_words)
        print('Unique words:', len(words))

        # Remove ignored words from set
        words = sorted(set(words) - ignored)
        print('Unique words after removeing ignored words:', len(words))

        # Create two dictionaries. One with word as a key and index as value. One with index as key and word as a value
        word_indices = dict((c, i) for i, c in enumerate(words))
        indices_word = dict((i, c) for i, c in enumerate(words))

        return word_indices, indices_word, text_in_words, ignored


def create_and_filter_sequences(text_in_words, ignored_words):
    STEP = 1
    sentences = []
    next_words = []
    ignored = 0

    # Loop original corpus. Add SEQUENCES_LEN long sentences to sentences and SEQUENCES_LEN next words to next_words
    # Only add sentences that don't contain ignored words
    for i in range(0, len(text_in_words) - SEQUENCE_LEN, STEP):
        # Only add sequences where no word is in ignored_words
        if len(set(text_in_words[i: i + SEQUENCE_LEN + 1]).intersection(ignored_words)) == 0:
            sentences.append(text_in_words[i: i + SEQUENCE_LEN])
            next_words.append(text_in_words[i + SEQUENCE_LEN])
        else:
            ignored = ignored + 1
    print('Ignored sequences:', ignored)
    print('Remaining sequences:', len(sentences))
    print('setnences ', sentences[0])
    print('next ', next_words[0])
    return sentences, next_words


def shuffle_and_split_training_set(sentences_original, next_original, percentage_test=2):
    # shuffle at unison
    print('Shuffling sentences')

    tmp_sentences = []
    tmp_next_word = []

    for i in np.random.RandomState(seed=42).permutation(len(sentences_original)):
        tmp_sentences.append(sentences_original[i])
        tmp_next_word.append(next_original[i])

    cut_index = int(len(sentences_original) * (1. - (percentage_test / 100.)))
    x_train, x_test = tmp_sentences[:cut_index], tmp_sentences[cut_index:]
    y_train, y_test = tmp_next_word[:cut_index], tmp_next_word[cut_index:]

    print("Size of training set = %d" % len(x_train))
    print("Size of test set = %d" % len(y_test))
    return (x_train, y_train), (x_test, y_test)


def save_sets(x_train, y_train, x_test, y_test):
    test = 'sad'
    return


if __name__ == '__main__':
    corpus_path = '../lyrics/raw.txt'
    word_indices, indices_word, text_in_words, ignored_words = corpus_to_dictionary(corpus_path)
    sequences, next_words = create_and_filter_sequences(text_in_words, ignored_words)
    (sentences_train, next_words_train), (sentences_test, next_words_test) = shuffle_and_split_training_set(sequences, next_words)
    save_sets(sentences_test, next_words_test, sentences_test, next_words_test)
