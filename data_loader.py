"""Read the data set and load the data in memory"""

import pandas as pd
import warnings
import logging
import os
import urllib.request as urlreq


class DataLoader:
    def __init__(self, file_directory, file_name, download_path, **params):
        """
        this class contains operations for loading data

        :param data_path: local path to the data
        :param download_path: if local path does not exist, download the data from the download_path
        :param params: other params when loading data, these params have to be accepted by pandas.read_csv()
        """

        self.file_directory = file_directory
        self.file_name = file_name
        self.download_path = download_path
        self.data_path = self.get_data_path()
        self.params = params

    def get_data_path(self):
        """get file path

        return: data_path
        """
        if not os.path.isdir(self.file_directory):
            print("Warning: File Directory {} does not exist, try to create it".format(self.file_directory))
            os.mkdir(self.file_directory)
            print("Successfully created {}".format(self.file_directory))

        data_path = self.file_directory + self.file_name

        if not os.path.exists(data_path):
            print("Warning: {0} does not exist, try to download data from {1}".format(data_path, self.download_path))
            self.download_data(data_path)

        return data_path

    def download_data(self, data_path):
        """download data if local data_path does not exist

        :return None
        """
        try:
            file_data = urlreq.urlopen(self.download_path)
            data_to_write = file_data.read()

            with open(data_path, 'wb') as f:
                f.write(data_to_write)
            print("Successfully downloaded data from {0} and store at {1}".format(self.download_path, data_path))

        except Exception as e:
            print("Downloading data failed. The error is: {0}".format(str(e)))
            exit(-1)

    def load_data(self):
        """ Load data: first try load data from local_path, if not exist, try download_path. If either works, issue error

        :return: None
        """
        try:
            self.df = pd.read_csv(self.data_path, **self.params)
            print(
                "Successfully loaded the data from {0} with additional params {1}".format(self.data_path, self.params))
        except Exception as e:
            print("Loading the data failed. The error is : {0}".format(str(e)))

        return df
