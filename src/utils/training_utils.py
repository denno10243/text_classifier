"""Utils for training."""

import pandas as pd


def resample_data(
    df: pd.DataFrame, class_col: str, upsample: bool = False
) -> pd.DataFrame:
    """Up or downsamples data in dataframe.
    
    Data is downsampled such that all classes have the same number of samples.
    The all classes will have the same number of samples as the class with the
    smallest number of samples
    
    Data is upsampled such that all classes have the same number of samples.
    The all classes will have the same number of samples as the class with the
    largest number of samples
    
    Args:
        df: dataframe containing training data
        class_col: column containing class for each data sample
        upsample: upsample the data (True), or downsample the data (False, default)

    Returns:
        df_train: updated dataframe containing up/downsampled training data
    """
    # get the value counts for each class
    vc = df[class_col].value_counts()
    if upsample:
        # get the largest number of samples
        n_samples = vc.max()
    else:
        # get the smallest number of samples
        n_samples = vc.min()
    # initially set df_train = None
    df_train = None
    # loop through all classes
    for d in vc.index:
        if df_train is not None:
            # if df_train already contains data, concatenate data for class = d
            # sample n_samples samples
            df_train = pd.concat(
                [
                    df_train.copy(deep=True),
                    df.loc[df[class_col] == d, :]
                    .copy(deep=True)
                    .sample(n=n_samples, replace=upsample),
                ]
            )
        else:
            # if df_train does not have data, set to class = d
            # sample n_samples samples
            df_train = (
                df.loc[df[class_col] == d, :]
                .copy(deep=True)
                .sample(n=n_samples, replace=upsample)
            )
    return df_train
