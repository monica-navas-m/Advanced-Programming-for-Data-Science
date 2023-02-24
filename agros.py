"""
The OS module in Python provides a way of using operating system dependent functionality.
"""
import os
import urllib.request
import pandas as pd


class Agros:
    """
    A class for downloading and processing agricultural data.
    """

    def __init__(self):
        """
        Initializes the Agros class.
        """
        self.url = "https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/Agricultural%20total%20factor%20productivity%20(USDA)/Agricultural%20total%20factor%20productivity%20(USDA).csv"
        self.filename = "Agricultural total factor productivity (USDA).csv"
        self.download_dir = os.path.join(os.getcwd(), "downloads")
        self.download_path = os.path.join(self.download_dir, self.filename)
        self.dataset = None

    def download_data(self):
        """
        Downloads data from a given URL and saves it to a specified file path.

        Returns:
            pandas.DataFrame: The downloaded dataset.
        """
        if os.path.isfile(self.download_path):
            print("Data file already exists, skipping download...")
            self.dataset = pd.read_csv(
                "downloads/Agricultural total factor productivity (USDA).csv"
            )
            return self.dataset

        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)
            print(f"Created directory: {self.download_dir}")

        print(f"Downloading data file from {self.url}...")
        urllib.request.urlretrieve(self.url, self.download_path)

        print(f"Data file downloaded and saved to {self.download_path}")
        self.dataset = pd.read_csv(
            "downloads/Agricultural total factor productivity (USDA).csv"
        )
        return self.dataset

<<<<<<< HEAD
    def get_countries(self):
        """
        Returns a list of available countries in the dataset.

        Returns
        -------
        list
            A list of strings representing the available countries in the dataset.
        """
        if self.dataset is None:
            self.download_data()

        return list(self.dataset["Entity"].unique())
=======
    def plot_correlation(self):

        """
        Plots a heatmap of the correlations between the specified columns.

        Parameters
        ----------
        columns : list of str
            A list of strings representing the columns to include in the correlation analysis.

        Returns
        -------
        None
            This function has no return value.
        """

        if self.dataset is None:
            self.download_data()

        columns = [
            "output_quantity",
            "crop_output_quantity",
            "animal_output_quantity",
            "fish_output_quantity",
            "ag_land_quantity",
            "labor_quantity",
            "capital_quantity",
            "machinery_quantity",
            "livestock_quantity",
            "fertilizer_quantity",
            "animal_feed_quantity",
            "cropland_quantity",
            "pasture_quantity",
            "irrigation_quantity",
        ]

        corr = self.dataset[columns].corr()
        sns.heatmap(corr, cmap="coolwarm", annot=True)
>>>>>>> 1b55288 (included the correlation plot method)
