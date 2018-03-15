import errno
import os

import numpy

from tools import load_cache_word_vectors, write_cache_word_vectors


def load_word_vectors(file, dim):
    """
    Read the word vectors from a text file
    Args:
        file (): the filename
        dim (): the dimensions of the word vectors

    Returns:
        word2idx (dict): dictionary of words to ids
        idx2word (dict): dictionary of ids to words
        embeddings (numpy.ndarray): the word embeddings matrix

    """
    # in order to avoid this time consuming operation, cache the results
    try:
        cache = load_cache_word_vectors(file)
        print("Loaded word embeddings from cache.")
        return cache
    except FileNotFoundError:
        pass

    # create the necessary dictionaries and the word embeddings matrix
    if os.path.exists(file):
        print('Indexing file {} ...'.format(file))

        word2idx = {}  # dictionary of words to ids
        idx2word = {}  # dictionary of ids to words
        embeddings = []  # the word embeddings matrix

        # create the 2D array, which will be used for initializing
        # the Embedding layer of a NN.
        # We reserve the first row (idx=0), as the word embedding,
        # which will be used for zero padding (word with id = 0).
        embeddings.append(numpy.zeros(dim))

        # read file, line by line
        with open(file, "r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                values = line.split(" ")
                word = values[0]
                vector = numpy.asarray(values[1:], dtype='float32')

                idx2word[i] = word
                word2idx[word] = i
                embeddings.append(vector)

            # add an unk token, for OOV words
            if "<unk>" not in word2idx:
                idx2word[len(idx2word) + 1] = "<unk>"
                word2idx["<unk>"] = len(word2idx) + 1
                embeddings.append(
                    numpy.random.uniform(low=-0.05, high=0.05, size=dim))

            print('Found %s word vectors.' % len(embeddings))
            embeddings = numpy.array(embeddings, dtype='float32')

        # write the data to a cache file
        write_cache_word_vectors(file, (word2idx, idx2word, embeddings))

        return word2idx, idx2word, embeddings

    else:
        print("{} not found!".format(file))
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), file)
