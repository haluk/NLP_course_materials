#!/usr/bin/env python3
import preprocess

TRAIN_FOLDER = "train/"
TEST_FOLDER = "test/"
POSITIVE_FOLDER = "pos/"
NEGATIVE_FOLDER = "neg/"


def text_decoder(text):
    """Decodes to utf-8"""
    return text.decode()


def truncate_to(text, pos=1000):
    """Show text until the character at pos index"""
    return text[:pos]


def archive_file_contents(tar, info):
    """Get contents of tar file"""
    f = tar.extractfile(info)
    return f.read()


def get_raw_data_from(tar):
    """Reads in the tar archive, forms the training and test set"""

    train_reviews = []
    train_labels = []
    test_reviews = []
    test_labels = []

    # for each file in the archive,
    # get the filename and tarinfo of the compressed file
    for fname, farchive in zip(tar.getnames(), tar.getmembers()):
        # if the file is in the training set
        if TRAIN_FOLDER in fname:

            # and a positive review
            if POSITIVE_FOLDER in fname:
                # add the review to the train_reviews list
                train_reviews.append(text_decoder(archive_file_contents(tar, farchive)))
                # add a 1 to the train_labels list indicating the example is a positive review
                train_labels.append(1)

            # if the file is a negative review
            if NEGATIVE_FOLDER in fname:
                # add the review to the train_reviews list
                train_reviews.append(text_decoder(archive_file_contents(tar, farchive)))
                # add a 0 to the train_labels list indicating the example is a negative review
                train_labels.append(0)

        # if the file is in the test set
        elif TEST_FOLDER in fname:

            # and a positive review
            if POSITIVE_FOLDER in fname:
                # add the review to the test_reviews list
                test_reviews.append(text_decoder(archive_file_contents(tar, farchive)))
                # add a 1 to the test_labels list indicating the example is a positive review
                test_labels.append(1)

            # if the file is a negative review
            if NEGATIVE_FOLDER in fname:
                # add the review to the test_reviews list
                test_reviews.append(text_decoder(archive_file_contents(tar, farchive)))
                # add a 0 to the test_labels list indicating the example is a negative review
                test_labels.append(0)

    return train_reviews, train_labels, test_reviews, test_labels


def tokenizer(doc):
    doc = preprocess.clean_text(doc)

    return preprocess.lemmatize(doc)
