#!/usr/bin/env python3
import re
import unicodedata

import contractions
import nltk
import spacy
from bs4 import BeautifulSoup
from nltk.tokenize.toktok import ToktokTokenizer

nlp = spacy.load("en_core_web_sm")
tokenizer = ToktokTokenizer()
stopword_list = nltk.corpus.stopwords.words("english")

# reference: https://github.com/dyn1990/YelpTopicWebApp/blob/ae205514f8a39c1676df094c76044c3fae904369/TextClean.py


def strip_html_tags(text):
    """strip HTML, this removes all the html formats and leave the content

    >>> strip_html_tags("<div, class='myclass'>my text.</div>")
    "my text."
    """
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text()

    return stripped_text


def remove_accented_chars(text):
    """this function translate unicode to ascii code, then decode it into utf

    >>> remove_accented_chars(u'Málaga')
    "Malaga"
    """
    text = (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("utf-8", "ignore")
    )

    return text


def expand_contractions(text):

    """this function expand the contracted words
    >>> expand_contractions("I don't think she's here")

    "I do not think she is here"
    """
    return contractions.fix(text)


def remove_url(text):
    return "".join(
        re.sub("(http[s]:\/\/)?(www\.)?\S*[.][a-zA-Z]{2,4}(\/)?\S+(\/)?", "", text)
    )


def remove_tweet(text):
    return "".join(re.sub("(@[\w]+)|([Rr][Tt])", " ", text))


def remove_special_characters(text):
    """this function removes punctuations
    """
    text = re.sub("[^a-zA-Z0-9\s]", " ", text)

    return text


def lemmatize(text):
    """this function lemmalize text

    >>> lemmatize_text("he has not been discovered")
    "he has not been discovered"
    """
    text = nlp(text)

    return [word.lemma_ if word.lemma_ != "-PRON-" else word.text for word in text]


def remove_stopwords(text, is_lower_case=False):
    """
    >>> remove_stopwords("he has not been discovered")
    "not discovered"
    """
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [
            token for token in tokens if token.lower() not in stopword_list
        ]
    filtered_text = " ".join(filtered_tokens)

    return filtered_text


def clean_text(
    doc,
    html_stripping=True,
    contraction_expansion=True,
    accented_char_removal=True,
    special_char_removal=True,
    stopword_removal=True,
    url_removal=True,
    tweet_removal=True,
    text_lower_case=True,
):

    if html_stripping:
        doc = strip_html_tags(doc)

    if accented_char_removal:
        doc = remove_accented_chars(doc)

    if contraction_expansion:
        doc = expand_contractions(doc)

    if text_lower_case:
        doc = doc.lower()

    if url_removal:
        doc = remove_url(doc)

    if tweet_removal:
        doc = remove_tweet(doc)

    # remove extra newlines
    doc = re.sub(r"[\r|\n|\r\n]+", " ", doc)

    if special_char_removal:
        doc = remove_special_characters(doc)

    # remove extra whitespace
    doc = re.sub(" +", " ", doc)

    if stopword_removal:
        doc = remove_stopwords(doc, is_lower_case=text_lower_case)

    return doc


def clean_corpus(
    corpus,
    html_stripping=True,
    contraction_expansion=True,
    accented_char_removal=True,
    text_lemmatization=True,
    special_char_removal=True,
    stopword_removal=True,
    url_removal=True,
    tweet_removal=True,
    text_lower_case=True,
):

    cleaned_corpus = []

    for doc in corpus:
        cleaned_corpus.append(clean_text(doc))

    return cleaned_corpus
