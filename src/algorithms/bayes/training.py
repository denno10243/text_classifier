"""Trains naive bayes text classification model."""

import copy
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy
from sklearn.naive_bayes import MultinomialNB

from src.processing.process_text import (get_stop_words, lemmatize_text,
                                         remove_stop_words, singularize_text, stem_text,
                                         vectorize_text)
from src.utils.training_utils import resample_data


def train_naive_bayes(
    df: pd.DataFrame,
    sample_col: str,
    class_col: str,
    alpha: float = 0.3,
    n_splits: int = 5,
    n_iter: int = 5,
    upsample: bool = None,
    use_stop_words: bool = True,
    use_lemmatizer: bool = False,
    use_stemmer: bool = False,
    use_singularizer: bool = False,
    tf_idf: bool = False,
) -> Tuple[pd.DataFrame, Tuple[bool, bool, bool, bool, bool, List, List]]:
    """
    Trains a classifier on dataframe of text.
    
    Uses cross-validation, splitting the data into train and test sets, with classifier
    fitted to the train split and predictions made on the test split.
    
    Process is repeated n_iter times, with the mode of the predictions taken as the final
    prediction.
    
    Args:
        df: dataframe containing lines
        sample_col: column containing the text sample to be classified 
        class_col: column containing class for each data sample
        alpha: additive smoothing parameter, 0 for no smoothing
        n_splits: number of splits to use for cross-validation
        n_iter: number of times the cross-validation is repeatted
        upsample: upsample the train data (True), downsample (False), or neither (None)
        use_stop_words: use stop words in the fit/predications
        use_lemmatizer: lemmatize the lines
        use_stemmer: stem the lines. If a lemmatizer is used, this is not used
        use_singularizer: singularize the words in each line
        tf_idf: use tf-idf vectorizer

    Returns:
        df: dataframe with final predicted class for each line and at each iteration added
        fitted_models: tuple used for later inference. Contains:
                       - flags used to process the data (use_stop_words etc.): 0 to 3
                       - vectorizers fitted to the: 4
                       - classifiers used to fit the data: 5
                       - list of stop words to use if removing them: 6

    """

    # if removing stop words, get a list of them
    if not use_stop_words:
        stop_words = get_stop_words()
    else:
        stop_words = None

    # initialize tuple to return info on fitted models
    fitted_models = (
        use_stop_words,
        use_lemmatizer,
        use_stemmer,
        use_singularizer,
        [],
        [],
        stop_words,
    )

    # go through n_iter iterations
    for i_iter in range(n_iter):

        # add lists to store vectorizers and classifiers for this iteration
        fitted_models[4].append([])
        fitted_models[5].append([])

        # initialize a column for predictions for this iteration
        df.loc[:, f"preds_NB_{i_iter + 1}"] = 0

        df_iter = process_data_iteration(
            df,
            sample_col,
            use_stop_words,
            use_singularizer,
            use_lemmatizer,
            use_stemmer,
            stop_words,
        )

        # split the indices into n_splits groups
        inds_all = np.array_split(df_iter.index, n_splits)
        # go through each group of indices, use that as the test split, and the
        # remaining groups as the training split
        for i, inds in enumerate(inds_all):
            # get the train and test sets
            df_train = df_iter.copy(deep=True).loc[~df_iter.index.isin(inds), :]
            df_test = df_iter.copy(deep=True).loc[df_iter.index.isin(inds), :]

            # up or downsample the training data
            if upsample is not None:
                df_train = resample_data(df_train, class_col, upsample)

            # get word count vectors for train and test sets
            (vectors_train, vectors_test, vectorizer) = vectorize_text(
                df_train, df_test, sample_col, tf_idf=tf_idf
            )

            # save the vectorizer
            fitted_models[4][-1].append(copy.copy(vectorizer))

            # initialize the model
            mnb = MultinomialNB(alpha=alpha)
            # fit to the training data
            mnb.fit(vectors_train, df_train[class_col].astype(int).values)
            # make predications on the test set
            test_pred = mnb.predict(vectors_test)
            # save results for iteration i_iter
            df.loc[inds, f"preds_NB_{i_iter + 1}"] = test_pred

            # save the fitted model
            fitted_models[5][-1].append(copy.copy(mnb))

        pred_col = f"preds_NB_{i_iter + 1}"
        print_progress(df, pred_col, class_col)

    # get the final prediction: the mode of all the predications
    cols = [f"preds_NB_{i_iter + 1}" for i_iter in range(n_iter)]
    df.loc[:, f"preds_NB_final"] = df[cols].apply(
        lambda x: scipy.stats.mode(x)[0][0], axis=1
    )

    # print the final result
    pred_col = f"preds_NB_final"
    print_progress(df, pred_col, class_col)

    return df, fitted_models


def process_data_iteration(
    df: pd.DataFrame,
    sample_col: str,
    use_stop_words: bool,
    use_singularizer: bool,
    use_lemmatizer: bool,
    use_stemmer: bool,
    stop_words: Optional[List] = None,
) -> pd.DataFrame:
    """Process the data for a training iteration.

    Args:
        df: dataframe containing unprocessed data
        sample_col: column containing text samples to process
        use_stop_words: keep stop words in text
        use_singularizer: singularize plurals
        use_lemmatizer: lemmatize words
        use_stemmer: stem words (only if not lemmatizing)
        stop_words: list of stopwords, or None if not used

    Returns:
        df_iter: dataframe containing processed data
    """

    # shuffle the data
    df_iter = df.copy(deep=True).sample(frac=1.0)

    # preprocess the data
    if not use_stop_words:
        # remove stop words
        df_iter = remove_stop_words(df_iter, sample_col, stop_words)
    # singularize
    if use_singularizer:
        df_iter = singularize_text(df_iter, sample_col)
    # lemmatize or stem
    if use_lemmatizer:
        # lemmatize the data
        df_iter = lemmatize_text(df_iter, sample_col)
    elif use_stemmer:
        # stem the data
        df_iter = stem_text(df_iter, sample_col)

    return df_iter


def print_progress(df: pd.DataFrame, pred_col: str, class_col: str) -> None:
    """Outputs the training progress.

    Args:
        df: dataframe containing predictions
        pred_col: prediction column to use
        class_col: column containing class for each data sample
    """

    acc = 100.0 * sum(df[pred_col] == df[class_col]) / len(df)
    acc_class = []
    for c in sorted(df[class_col].unique()):
        df_sub = df.loc[df[class_col] == c, :].copy(deep=True)
        acc_sub = 100.0 * sum(df_sub[pred_col] == df_sub[class_col]) / len(df_sub)
        acc_class.append(acc_sub)
    acc_sub_str = ", ".join([f"{round(a, 2)}%" for a in acc_class])
    print(pred_col.ljust(40) + f"{round(acc, 2)}%".ljust(10) + f"[{acc_sub_str}]")
