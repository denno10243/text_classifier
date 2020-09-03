"""Trains a user selected model."""

import argparse
import json

import pandas as pd

from src.algorithms.bayes.naive_bayes import NaiveBayes

if __name__ == "__main__":
    # initialize the arg parser
    parser = argparse.ArgumentParser()
    # get the arguments
    parser.add_argument(
        "-c", "--config_path", default=None, type=str, help="Enter path for config file"
    )
    args = parser.parse_args()

    # load the config
    with open(args.config_path, "r") as conf:
        config = json.load(conf)

    # initialize the model
    nb = NaiveBayes(config["training_config"])

    # read in the data
    df_train = pd.read_csv(config["training_data_path"])
    # fit the model
    nb.train(df_train["sample"].values, df_train["label"].values)
    # save the model
    nb.save_model(config["model_output_fname"])
