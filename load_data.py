import glob
import html
import os

SEPARATOR = "\t"


def clean_text(text):
    """
    Remove extra quotes from text files and html entities
    Args:
        text (str): a string of text

    Returns: (str): the "cleaned" text

    """
    text = text.rstrip()

    if '""' in text:
        if text[0] == text[-1] == '"':
            text = text[1:-1]
        text = text.replace('\\""', '"')
        text = text.replace('""', '"')

    text = text.replace('\\""', '"')

    text = html.unescape(text)
    text = ' '.join(text.split())
    return text


def parse_file(file):
    """
    Read a file and return a dictionary of the data, in the format:
    tweet_id:{topic,sentiment, text}
    """

    data = {}
    lines = open(file, "r", encoding="utf-8").readlines()
    for line_id, line in enumerate(lines):
        columns = line.rstrip().split(SEPARATOR)
        if columns[2] == "positive" or columns[2] == "negative":
            tweet_id = columns[0]
            topic = columns[1]
            sentiment = columns[2]
            text = columns[3:]
            text = clean_text(" ".join(text))
            data[tweet_id] = (topic, sentiment, text)
        elif columns[2] == "-2" or columns[2] == "-1" or columns[2] == "0" or columns[2] == "1" or columns[2] == "2":
            tweet_id = columns[0]
            topic = columns[1]
            sentiment = columns[2]
            text = columns[3:]
            text = clean_text(" ".join(text))
            data[tweet_id] = (topic, sentiment, text)
    return data


def load_data_from_dir(path):
    FILE_PATH = os.path.dirname(__file__)
    files_path = os.path.join(FILE_PATH, path)

    files = glob.glob(files_path + "/**/*.tsv", recursive=True)
    files.extend(glob.glob(files_path + "/**/*.txt", recursive=True))

    data = {}  # use dict, in order to avoid having duplicate tweets (same id)
    for file in files:
        file_data = parse_file(file)
        data.update(file_data)
    return list(data.values())

