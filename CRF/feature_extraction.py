import spacy

MODEL = spacy.load('de_core_news_sm')

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

def tokenize_document(text, include_stop=False):
    '''

    :param text: A document
    :param include_stop: To include stop words or not
    :return: A list of tokens of the document
    '''
    global MODEL
    doc = MODEL(text)
    sents = []
    for sent in doc.sents:
        sents.append([word.text for word in sent if (not word.is_stop and not include_stop)])
    return sents

def sent2features(sent):
    '''
    :param sent: A list of words in a sentence
    :return: List of dictionaries. each dict has features for word
    '''
    return [word2features(sent, i) for i in range(len(sent))]


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

def sent2features(sent):
    '''

    :param sent:
    :return:
    '''
    return [word2features(sent, i) for i in range(len(sent))]