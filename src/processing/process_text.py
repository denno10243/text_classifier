"""Functions for processing comment text."""

import glob
import os
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer


def get_stop_words() -> List:
    """Returns a set of stop words for the english language.

    Returns:
        sorted list of stop words
    """
    return sorted(set(stopwords.words("english")))


def remove_stop_words(
    df: pd.DataFrame, sample_col: str, stop_words: List
) -> pd.DataFrame:
    """Removes the stop words from the comments in a dataframe.
    
    Args:
        df: dataframe containing comments
        sample_col: column containing text samples
        stop_words: list of stop words to remove

    Returns:
        df: dataframe containing comments with stop words removed
    """

    # remove stop words
    df.loc[:, sample_col] = (
        df.loc[:, sample_col]
        .copy(deep=True)
        .str.lower()
        .apply(word_tokenize)
        .apply(lambda x: " ".join([word for word in x if word not in stop_words]))
    )
    return df


def lemmatize_text(df: pd.DataFrame, sample_col: str) -> pd.DataFrame:
    """Applies a word net lemmatizer to the comments in a dataframe.
    
    Args:
        df: dataframe containing comments
        sample_col: column containing text samples

    Returns:
        df: dataframe containing comments with lemmatizer applied
    """
    # set lemmatizer
    lemmatizer = WordNetLemmatizer()
    # apply to the text
    df.loc[:, sample_col] = (
        df[sample_col].copy(deep=True).str.lower().apply(lemmatizer.lemmatize)
    )
    return df


def stem_text(df: pd.DataFrame, sample_col: str) -> pd.DataFrame:
    """Applies a porter stemmer to the comments in a dataframe.
    
    Args:
        df: dataframe containing comments
        sample_col: column containing text samples

    Returns:
        df: dataframe containing comments with stemmer applied
    """
    # set stemmer
    ps = PorterStemmer()
    # apply to the text
    df.loc[:, sample_col] = df[sample_col].copy(deep=True).str.lower().apply(ps.stem)
    return df


def singularize_text(df: pd.DataFrame, sample_col: str) -> pd.DataFrame:
    """Singularizes text.
    
    Args:
        df: dataframe containing comments
        sample_col: column containing text samples

    Returns:
        df: dataframe containing singularized comments
    """
    # apply to the text
    df.loc[:, sample_col] = (
        df[sample_col].copy(deep=True).str.lower().apply(singularize)
    )
    return df


def vectorize_text(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    sample_col: str,
    tf_idf: bool = False,
) -> Tuple[np.array, np.array, CountVectorizer]:
    """Vectorizes text, returning a word count for each comment.

    Vector is fitted to the training data, and applied to the training and test
    data.
    
    Args:
        df_train: dataframe containing training data
        df_test: dataframe containing test data
        sample_col: column containing text samples
        tf_idf: use term frequency, inverse document frequency

    Returns
        vectors_train: array containing vector of word counts for each comment
                       in the training set
        vectors_test: array containing vector of word counts for each comment
                      in the test set
        vectorizer: fitted count vectorizer
    """
    # create the transform
    if tf_idf:
        vectorizer = TfidfVectorizer()
    else:
        vectorizer = CountVectorizer()
    # fit to the training data
    vectorizer.fit(df_train[sample_col])
    # encode train and test set
    vector_train = vectorizer.transform(df_train[sample_col])
    vector_test = vectorizer.transform(df_test[sample_col])
    return vector_train, vector_test, vectorizer
