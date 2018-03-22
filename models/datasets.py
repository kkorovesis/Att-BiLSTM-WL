from ekphrasis.classes.preprocessor import TextPreProcessor
from ekphrasis.classes.tokenizer import SocialTokenizer
from ekphrasis.dicts.emoticons import emoticons
from sklearn import preprocessing
from torch.utils.data import Dataset
from collections import Counter
from load_data import load_data_from_dir
from nlp import vectorize


class SentimentDataset(Dataset):
    def __init__(self, file, max_length, max_topic_length, word2idx, tword2idx, topic_bs):
        """
        A PyTorch Dataset
        What we have to do is to implement the 2 abstract methods:

            - __len__(self): in order to let the DataLoader know the size
                of our dataset and to perform batching, shuffling and so on...
            - __getitem__(self, index): we have to return the properly
                processed data-item from our dataset with a given index

        Args:
            file (str): path to the data file
            max_length (int): the max length for each sentence.
                if 0 then use the maximum length in the dataset
            word2idx (dict): a dictionary which maps words to indexes
        """

        self.text_processor = TextPreProcessor(
            # terms that will be normalized
            normalize=['url', 'email', 'percent', 'money', 'phone', 'user',
                       'time', 'url', 'date', 'number'],
            # terms that will be annotated
            annotate={"hashtag", "allcaps", "elongated", "repeated",
                      'emphasis', 'censored'},
            fix_html=True,  # fix HTML tokens

            # corpus from which the word statistics are going to be used
            # for word segmentation
            segmenter="twitter",

            # corpus from which the word statistics are going to be used
            # for spell correction
            corrector="twitter",

            unpack_hashtags=True,  # perform word segmentation on hashtags
            unpack_contractions=True,  # Unpack contractions (can't -> can not)
            spell_correct_elong=False,  # spell correction for elongated words

            # select a tokenizer. You can use SocialTokenizer, or pass your own
            # the tokenizer, should take as input a string and return a list of tokens
            tokenizer=SocialTokenizer(lowercase=True).tokenize,

            # list of dictionaries, for replacing tokens extracted from the text,
            # with other expressions. You can pass more than one dictionaries.
            dicts=[emoticons]
        )


        self.word2idx = word2idx
        self.tword2idx = tword2idx

        print("loading dataset from {}...".format(file))
        _data = load_data_from_dir(file)
        if topic_bs:
            self.data = [x[2] for x in _data]
            self.labels = [x[1] for x in _data]
            self.topics = [x[0] for x in _data]
        else:
            self.data = [x[1] for x in _data]
            self.labels = [x[0] for x in _data]


        print("Tokenizing...")
        # self.data = [tokenize(x) for x in self.data]
        self.data = [self.text_processor.pre_process_doc(x) for x in self.data]
        self.topics = [self.text_processor.pre_process_doc(x) for x in self.topics]

        # if max_length == 0, then set max_length
        # to the maximum sentence length in the dataset
        if max_length == 0:
            self.max_length = max([len(x) for x in self.data])
        else:
            self.max_length = max_length

        if max_topic_length == 0:
            self.max_topic_length = max([len(x) for x in self.topics])
        else:
            self.max_topic_length = max_topic_length

        # define a mapping for the labels,
        # for transforming the string labels to numbers
        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder = self.label_encoder.fit(self.labels)

        self.label_count = Counter(self.labels)
        self.weights = [self.label_count['-1'], self.label_count['2'],
                        self.label_count['0'], self.label_count['1'],
                        self.label_count['2']]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns the _transformed_ item from the dataset

        Args:
            index (int):

        Returns:
            (tuple):
                * example (ndarray): vector representation of a training example
                * label (string): the class label
                * length (int): the length (tokens) of the sentence
                * index (int): the index of the returned dataitem in the dataset.
                  It is useful for getting the raw input for visualizations.

        Examples:
            For an `index` where:
            ::
                self.data[index] = ['super', 'eagles', 'coach', 'sunday', 'oliseh',
                                    'meets', 'with', 'chelsea', "'", 's', 'victor',
                                    'moses', 'in', 'london', '<url>']
                self.target[index] = "neutral"

            the function will return:
            ::
                example = [  533  3908  1387   649 38127  4118    40  1876    63   106  7959 11520
                            22   888     7     0     0     0     0     0     0     0     0     0
                             0     0     0     0     0     0     0     0     0     0     0     0
                             0     0     0     0     0     0     0     0     0     0     0     0
                             0     0]
                label = 1
        """

        sample, label, topic = self.data[index], self.labels[index], self.topics[index]

        # transform the sample and the label,
        # in order to feed them to the model
        message = vectorize(sample, self.word2idx, self.max_length)
        topic = vectorize(topic, self.tword2idx, self.max_topic_length)
        label = self.label_encoder.transform([label])[0]

        return message, topic, label, len(self.data[index]), len(self.topics[index]), self.weights, index
