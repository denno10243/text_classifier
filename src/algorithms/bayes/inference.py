"""Carries out inference using previously fitted models."""

from typing import List, Optional, Tuple

import pandas as pd
import scipy

from src.processing.process_text import (
    get_stop_words,
    lemmatize_text,
    remove_stop_words,
    singularize_text,
    stem_text,
    vectorize_text,
)


def inference_naive_bayes(
    df: pd.DataFrame,
    fitted_models: Tuple[bool, bool, bool, bool, List, List, Optional[List]],
    sample_col: str,
    class_col: str,
) -> pd.DataFrame:
    """Performs inference using previously fitted models.
    
    Args:
        df: dataframe containing data to cary out inference on
        fitted_models: tuple with previously fitted models. Contains:
                       - flags used to process the data (use_stop_words etc.): 0 to 3
                       - vectorizers fitted to the: 4
                       - classifiers used to fit the data: 5
                       - list of stop words to use if removing them: 6
        sample_col: column containing text sample to make prediction on
        class_col: column containing class that is being prediced for each data sample.

    Returns:
        df: dataframe with predictions from fitted models
    """

    # get number of iterations and number of splits that were used
    n_iter = len(fitted_models[4])
    n_splits = len(fitted_models[4][-1])

    # get flags
    use_stop_words = fitted_models[0]
    use_lemmatizer = fitted_models[1]
    use_stemmer = fitted_models[2]
    use_singularizer = fitted_models[3]

    # preprocess the data
    if not use_stop_words:
        # remove stop words
        stop_words = fitted_models[6]
        df = remove_stop_words(df, sample_col, stop_words)
    # singularize
    if use_singularizer:
        df = singularize_text(df, sample_col)
    # lemmatize or stem
    if use_lemmatizer:
        # lemmatize the data
        df = lemmatize_text(df, sample_col)
    elif use_stemmer:
        # stem the data
        df = stem_text(df, sample_col)

    for i_iter, (vectorizers, models) in enumerate(
        zip(fitted_models[4], fitted_models[5])
    ):
        for i_split, (vectorizer, model) in enumerate(zip(vectorizers, models)):
            # transform the data based on the vectorizer
            vectors = vectorizer.transform(df[sample_col])
            # make predications on the test set
            preds = model.predict(vectors)
            # initialize a column for predictions for this iteration and split
            df[f"preds_NB_{i_iter + 1}_{i_split + 1}"] = preds
            # print sub result
            print_progress(df, f"preds_NB_{i_iter + 1}_{i_split + 1}", class_col)

    # get the final prediction: the mode of all the predications
    cols = [
        f"preds_NB_{i_iter + 1}_{i_split + 1}"
        for i_iter in range(n_iter)
        for i_split in range(n_splits)
    ]
    df[f"preds_NB_final"] = df[cols].apply(lambda x: scipy.stats.mode(x)[0][0], axis=1)
    # print final result
    print_progress(df, f"preds_NB_final", class_col)
    return df


def print_progress(df: pd.DataFrame, pred_col: str, class_col: str) -> None:
    """Outputs the training progress.

    Args:
        df: dataframe containing predictions
        pred_col: prediction column to use
        class_col: column containing class for each data sample
    """

    if class_col in df.columns:
        acc = 100.0 * sum(df[pred_col] == df[class_col]) / len(df)
        acc_class = []
        for c in sorted(df[class_col].unique()):
            df_sub = df.loc[df[class_col] == c, :].copy(deep=True)
            acc_sub = 100.0 * sum(df_sub[pred_col] == df_sub[class_col]) / len(df_sub)
            acc_class.append(acc_sub)
            acc_sub_str = ", ".join([f"{round(a, 2)}%" for a in acc_class])
        print(pred_col.ljust(40) + f"{round(acc, 2)}%".ljust(10) + f"[{acc_sub_str}]")
    else:
        print(pred_col.ljust(40))
