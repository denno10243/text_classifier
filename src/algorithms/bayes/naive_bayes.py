"""Trains naive bayes text classification model."""

import copy
import pickle
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import scipy
from sklearn.naive_bayes import MultinomialNB

from src.processing.process_text import (
    get_stop_words,
    lemmatize_text,
    remove_stop_words,
    singularize_text,
    stem_text,
    vectorize_text,
)
from src.utils.training_utils import resample_data


class NaiveBayes:
    """Naive Bayes classifier

    Args:
        config: configuration for training, can contain the following.
                  - alpha: additive smoothing parameter, 0 for no smoothing -> [0.3]
                  - n_splits: number of splits to use for cross-validation -> [5]
                  - n_iter: number of times the cross-validation is repeatted -> [5]
                  - upsample: upsample the train data (True), downsample (False), or
                              neither (None) -> [None]
                  - use_stop_words: use stop words in the fit/predications -> [True]
                  - use_lemmatizer: lemmatize the lines -> [False]
                  - use_stemmer: stem the lines. If a lemmatizer is used,
                                 this is not used -> [False]
                  - use_singularizer: singularize the words in each line -> [False]
                  - tf_idf: use tf-idf vectorizer -> [False]
                If these values are not present, default values (in square brakets) are used.

    Attributes:
        config: configuration for training.
        stop_words: list of stop words that are to be removed
        fitted_models: tuple of fitted models used for later inference. Contains:
                       - vectorizers fitted to the: 0
                       - classifiers used to fit the data: 1
    """

    def __init__(self, config: Dict = {}) -> None:
        """Class Initializer."""
        # Default config
        self.config = {
            "alpha": 0.3,
            "n_splits": 5,
            "n_iter": 5,
            "upsample": None,
            "use_stop_words": True,
            "use_lemmatizer": False,
            "use_stemmer": False,
            "use_singularizer": False,
            "tf_idf": False
        }
        # update the config with user provided values
        self.config.update(config)
        # initialize empty fitted_models
        self.fitted_models = ()
        

    def train(self, samples: np.array, labels: np.array, verbose: int = 1,) -> np.array:
        """
        Trains a classifier on labelled text samples.

        Uses cross-validation, splitting the data into train and test sets, with classifier
        fitted to the train split and predictions made on the test split.

        Process is repeated n_iter times, with the mode of the predictions taken as the final
        prediction.

        Args:
            samples: text samples
            labels: labels for each sample
            verbose: level of information to output:
                     - 0: no ouput
                     - 1: output final accurracy
                     - 2: output all accuracies (final at for each iteration)

        Returns:
            predictions: final predicted class for each sample
        """

        # make a dataframe from the data
        df = pd.DataFrame(columns=["sample", "label"], data=zip(samples, labels))

        # if removing stop words, get a list of them
        if not self.config["use_stop_words"]:
            self.stop_words = get_stop_words()
        else:
            self.stop_words = []

        # initialize tuple to return info on fitted models
        self.fitted_models = (
            [],
            [],
        )

        # go through n_iter iterations
        for i_iter in range(self.config["n_iter"]):

            # add lists to store vectorizers and classifiers for this iteration
            self.fitted_models[0].append([])
            self.fitted_models[1].append([])

            # initialize a column for predictions for this iteration
            df.loc[:, f"preds_NB_{i_iter + 1}"] = 0

            df_iter = self.process_data_iteration(df)

            # split the indices into n_splits groups
            inds_all = np.array_split(df_iter.index, self.config["n_splits"])
            # go through each group of indices, use that as the test split, and the
            # remaining groups as the training split
            for i, inds in enumerate(inds_all):
                # get the train and test sets
                df_train = df_iter.copy(deep=True).loc[~df_iter.index.isin(inds), :]
                df_test = df_iter.copy(deep=True).loc[df_iter.index.isin(inds), :]

                # up or downsample the training data
                if self.config["upsample"] is not None:
                    df_train = resample_data(df_train, "label", self.config["upsample"])

                # get word count vectors for train and test sets
                (vectors_train, vectors_test, vectorizer) = vectorize_text(
                    df_train, df_test, "sample", tf_idf=self.config["tf_idf"],
                )

                # save the vectorizer
                self.fitted_models[0][-1].append(copy.copy(vectorizer))

                # initialize the model
                mnb = MultinomialNB(alpha=self.config["alpha"])
                # fit to the training data
                mnb.fit(vectors_train, df_train["label"].astype(int).values)
                # make predications on the test set
                test_pred = mnb.predict(vectors_test)
                # save results for iteration i_iter
                df.loc[inds, f"preds_NB_{i_iter + 1}"] = test_pred

                # save the fitted model
                self.fitted_models[1][-1].append(copy.copy(mnb))

            if verbose == 2:
                pred_col = f"preds_NB_{i_iter + 1}"
                print_progress(df, pred_col, "label")

        # get the final prediction: the mode of all the predications
        cols = [f"preds_NB_{i_iter + 1}" for i_iter in range(self.config["n_iter"])]
        df.loc[:, f"preds_NB_final"] = df[cols].apply(
            lambda x: scipy.stats.mode(x)[0][0], axis=1
        )

        if verbose != 0:
            # print the final result
            pred_col = f"preds_NB_final"
            print_progress(df, pred_col, "label")

        return df[f"preds_NB_final"].values

    def predict(
        self, samples: np.array, labels: np.array = None, verbose: int = 1,
    ) -> pd.DataFrame:
        """Performs inference using previously fitted models on text samples.

        Args:
            samples: text samples to make predictions for
            labels: labels for each sample, for final accuracy result (if supplied)
            verbose: level of information to output:
                     - 0: no ouput
                     - 1: output final accurracy
                     - 2: output all accuracies (final at for each iteration)

        Returns:
            df: dataframe with predictions from fitted models
        """
        # check that the model has been trained
        assert self.fitted_models, "Train model first: use train method or restore previous model"

        # make a dataframe from the data
        if labels is not None:
            df = pd.DataFrame(columns=["sample", "label"], data=zip(samples, labels))
        else:
            df = pd.DataFrame(columns=["sample"], data=samples)

        # preprocess the data
        if not self.config["use_stop_words"]:
            # remove stop words
            df = remove_stop_words(df, "sample", self.stop_words)
        # singularize
        if self.config["use_singularizer"]:
            df = singularize_text(df, "sample")
        # lemmatize or stem
        if self.config["use_lemmatizer"]:
            # lemmatize the data
            df = lemmatize_text(df, "sample")
        elif self.config["use_stemmer"]:
            # stem the data
            df = stem_text(df, "sample")

        for i_iter, (vectorizers, models) in enumerate(
            zip(self.fitted_models[0], self.fitted_models[1])
        ):
            for i_split, (vectorizer, model) in enumerate(zip(vectorizers, models)):
                # transform the data based on the vectorizer
                vectors = vectorizer.transform(df["sample"])
                # make predications on the test set
                preds = model.predict(vectors)
                # initialize a column for predictions for this iteration and split
                df[f"preds_NB_{i_iter + 1}_{i_split + 1}"] = preds
                if verbose == 2:
                    # print sub result
                    print_progress(df, f"preds_NB_{i_iter + 1}_{i_split + 1}", "label")

        # get the final prediction: the mode of all the predications
        cols = [
            f"preds_NB_{i_iter + 1}_{i_split + 1}"
            for i_iter in range(self.config["n_iter"])
            for i_split in range(self.config["n_splits"])
        ]
        df[f"preds_NB_final"] = df[cols].apply(
            lambda x: scipy.stats.mode(x)[0][0], axis=1
        )

        if verbose != 0:
            # print final result
            print_progress(df, f"preds_NB_final", "label")
            return df[f"preds_NB_final"].values

    def process_data_iteration(self, df: pd.DataFrame,) -> pd.DataFrame:
        """Process the data for a training iteration.

        Args:
            df: dataframe containing unprocessed data
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
        if not self.config["use_stop_words"]:
            # remove stop words
            df_iter = remove_stop_words(df_iter, "sample", self.stop_words)
        # singularize
        if self.config["use_singularizer"]:
            df_iter = singularize_text(df_iter, "sample")
        # lemmatize or stem
        if self.config["use_lemmatizer"]:
            # lemmatize the data
            df_iter = lemmatize_text(df_iter, "sample")
        elif self.config["use_stemmer"]:
            # stem the data
            df_iter = stem_text(df_iter, "sample")

        return df_iter

    def save_model(self, fname: str) -> None:
        """Saves the object as a pickle file.

        Args:
            fname: filename to save object as.
        """

        with open(fname, "wb") as f:
            pickle.dump(self.__dict__, f, protocol=4)

    def restore_model(self, fname: str) -> None:
        """Restores the object from a pickle file.

        Args:
            fname: filename to restore object from.
        """

        with open(fname, "rb") as f:
            obj_attributes = pickle.load(f)

        self.config = obj_attributes["config"]
        self.stop_words = obj_attributes["stop_words"]
        self.fitted_models = obj_attributes["fitted_models"]


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
