import numpy


def tokenize(text, lowercase=True):
    if lowercase:
        text = text.lower()
    return text.split()


def vectorize(text, word2idx, max_length, unk_policy="random"):
    """
    Covert array of tokens, to array of ids, with a fixed length
    and zero padding at the end
    Args:
        text (): the wordlist
        word2idx (): dictionary of word to ids
        max_length ():
        unk_policy (): how to handle OOV words

    Returns: list of ids with zero padding at the end

    """
    words = numpy.zeros(max_length).astype(int)

    # trim tokens after max length
    text = text[:max_length]

    for i, token in enumerate(text):
        if token in word2idx:
            words[i] = word2idx[token]
        else:
            if unk_policy == "random":
                words[i] = word2idx["<unk>"]
            elif unk_policy == "zero":
                words[i] = 0

    return words
