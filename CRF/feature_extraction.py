import spacy
import json
from sklearn.utils import shuffle

#IMPORTANT: This model should be the same as the one used for annotations.
MODEL = spacy.load('de_core_news_sm')

def tokenize_document(text):
    '''

    :param text: A document
    :param include_stop: To include stop words or not
    :return: A list of tokens of the document
    '''
    global MODEL
    doc = MODEL(text)
    sents = []
    for sent in doc.sents:
        sents.append([word.text for word in sent])
    return sents

def sent2features(sent):
    '''
    :param sent: A list of words in a sentence
    :return: List of dictionaries. each dict has features for word
    '''
    return [word2features(sent, i) for i in range(len(sent))]


def doc2features(text):
    '''

    :param text: A document in the form of a string
    :return: The features for the document
    '''
    sent_list = tokenize_document(text)
    doc_feats = []
    for sent in sent_list:
        doc_feats += sent2features(sent)
    return doc_feats


def word2features(sent, i):
    """
    :param sent: A list of words
    :param i: index of a particular word in a sentence
    :return: A list of dictionaries, with each dict containing features for a word
    """
    word = sent[i]

    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit()
    }
    if i > 0:
        prev_word = sent[i-1]
        features.update({
            'prev_word.lower()': prev_word.lower(),
            'prev_word.istitle()': prev_word.istitle(),
            'prev_word.isupper()': prev_word.isupper()
        })
    else:
        features['BOS'] = True

    if i < len(sent)-1:
        next_word = sent[i+1]
        features.update({
            'prev_word.lower()': next_word.lower(),
            'prev_word.istitle()': next_word.istitle(),
            'prev_word.isupper()': next_word.isupper()
        })
    else:
        features['EOS'] = True

    return features


def train_test_split(train_test_split, data_path):
    '''

    :param train_test_split: Percentage of train vs test documents
    :param data_path: The JSON file where in the text documents and their annotaitons are configured
    :return: The splitted Train and Test documents.
    '''
    docs = json.load(open(data_path, "r"))
    docs = shuffle(docs)
    total_docs = len(docs)
    train_end_idx = int(total_docs * train_test_split[0])
    train_docs = docs[:train_end_idx]
    test_docs = docs[train_end_idx:]
    return train_docs, test_docs

def get_tagname_via_start_idx(start_idx, tags):

    for tag in tags:
        if start_idx == tag['start_idx']:
            return tag['entity']

    return  None

def doc2labels(text, tags):

    annotated_toks_start_ids = {tag['start_idx'] for tag in tags}

    global MODEL
    doc = MODEL(text)

    labels = []
    for sent in doc.sents:
        for word in sent:
            if word.idx in annotated_toks_start_ids:
                ent_name = get_tagname_via_start_idx(word.idx, tags)
                labels.append(ent_name)
            else:
                labels.append('OTHER')

    return labels


def construct_train_test_dataset(TRAIN_TEST_SPLIT, DATA_PATH):
    '''

    :param TRAIN_TEST_SPLIT:  Percentage of train vs test documents
    :param DATA_PATH: JSON file path containing annotations.
    :return: The train test
    '''
    train_docs, test_docs = train_test_split(TRAIN_TEST_SPLIT, DATA_PATH)
    X_train = [doc2features(doc['text']) for doc in train_docs]
    X_test = [doc2features(doc['text']) for doc in test_docs]

    y_train = [doc2labels(doc['text'], doc['tags']) for doc in train_docs]
    y_test = [doc2labels(doc['text'], doc['tags']) for doc in test_docs]


    return  X_train, X_test, y_train, y_test