"""
The OS module in Python provides a way of using operating system dependent functionality.
"""
import os
import urllib.request
from typing import List, Optional, Union
import pandas as pd
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt


class Agros:
    """
    A class for downloading and processing agricultural data.
    """

    def __init__(self):
        """
        Initializes the Agros class.
        """
        self.url = (
            "https://raw.githubusercontent.com/owid/owid-datasets/master/datasets/"
            "Agricultural%20total%20factor%20productivity%20(USDA)/"
            "Agricultural%20total%20factor%20productivity%20(USDA).csv"
        )
        self.filename = "Agricultural total factor productivity (USDA).csv"
        self.download_dir = os.path.join(os.getcwd(), "downloads")
        self.download_path = os.path.join(self.download_dir, self.filename)
        self.dataset = None

    def download_data(self):
        """
        Downloads data from a given URL and saves it to a specified file path.

        Returns
        -------
            pandas.DataFrame: The downloaded dataset.
        """

        self.world = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))

        if os.path.isfile(self.download_path):
            self.dataset = pd.read_csv(
                "downloads/Agricultural total factor productivity (USDA).csv"
            )
            return (self.dataset, self.world)

        if not os.path.exists(self.download_dir):
            os.makedirs(self.download_dir)
            print(f"Created directory: {self.download_dir}")

        print(f"Downloading data file from {self.url}...")
        urllib.request.urlretrieve(self.url, self.download_path)

        print(f"Data file downloaded and saved to {self.download_path}")
        self.dataset = pd.read_csv(
            "downloads/Agricultural total factor productivity (USDA).csv"
        )
        return (self.dataset, self.world)

        continents = [
            "Asia",
            "Africa",
            "North America",
            "South America",
            "Antarctica",
            "Europe",
            "Australia",
        ]
        self.dataset = self.dataset[~self.dataset["Entity"].isin(continents)]

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

    def plot_correlation(self) -> None:

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
        # load data
        if self.dataset is None:
            self.download_data()

        data_frame = self.dataset.filter(regex="_quantity")
        corr = data_frame.corr()
        sns.heatmap(corr, cmap="coolwarm", annot=True, annot_kws={"size": 8})

    def areachart_country_output(
        self, country: Optional[str] = "World", normalize: bool = False
    ) -> None:

        """
        Plots an area chart of the  "_output_" columns for a given country

        Parameters
        ---------------
        country: string
            Country selected to plot area chart of the outputs, if *NONE* or
            'World' should plot the sum for all *distinct* countries
        normalize: boolean
            If True, normalizes the output in relative terms: each year, output
             should always be 100%

        Raises
        -------
            ValueError(f'{country} is not a valid')

        Returns
        -------
            Area Chart: This function returns area chart with outputs of a country by year

        """

        # Load data
        if self.dataset is None:
            self.download_data()

        data_frame = self.dataset.filter(regex="_output_|Year|Entity")

        # Filter by country if specified
        if country is not None:
            if country.lower() == "world" or country is None:
                data_frame = data_frame.groupby("Year").sum().reset_index()
            else:
                data_frame = data_frame[data_frame["Entity"] == country.capitalize()]
                if data_frame.empty:
                    raise ValueError(f"Country '{country}' not found in dataset.")
                data_frame = data_frame.drop("Entity", axis=1)

        # Normalize if specified
        if normalize:
            data_frame.iloc[:, 1:] = (
                data_frame.iloc[:, 1:]
                .div(data_frame.iloc[:, 1:].sum(axis=1), axis=0)
                .multiply(100)
            )

        # Plot an area chart for the output columns
        sns.set_theme(palette="bright")
        plt.stackplot(
            data_frame["Year"],
            data_frame.iloc[:, 1:].values.T,
            labels=data_frame.iloc[:, 1:].columns,
        )
        plt.legend()
        plt.xlabel("Year")
        plt.title(
            f'Agricultural Outputs {"for " + country if country else "All Countries"}'
        )
        plt.show()

    def plot_country_output(self, countries: Union[str, List[str]] = "World") -> None:

        """
        Plots the total output of one or more countries over time.

        Parameters
        ----------
        countries : str or list of str
            The name(s) of the country(ies) to plot.

        Returns
        -------
        None
        """

        # load data
        if self.dataset is None:
            self.download_data()

        data_frame = self.dataset

        if isinstance(countries, str):
            countries = [countries]

        # check if all specified countries are in dataset
        for country in countries:
            if country not in data_frame["Entity"].unique():
                raise ValueError(f"Country '{country}' not found in dataset.")

        # plot data
        fig, ax_plot = plt.subplots()
        for country in countries:
            country_data = data_frame[data_frame["Entity"] == country]
            ax_plot.plot(country_data["Year"], country_data["output"], label=country)

        ax_plot.legend()
        ax_plot.set_xlabel("Year")
        ax_plot.set_ylabel("Output")
        ax_plot.set_title("Comparison of Total Output by Country")

        plt.show()

    def gapminder(self, year: int) -> None:

        """
        Creates a scatter plot of fertilizer quantity vs. output quantity
        for a specific year in the Gapminder dataset, where the area of each
        dot represents the TFP (total factor productivity) for the respective year.

        Parameters
        ----------
        year : int
            The year for which to create the scatter plot.

        Raises
        ------
        TypeError
            If the year argument is not an integer.

        Returns
        -------
        None

        Notes
        -----
        This method assumes that the dataset has already been downloaded and loaded
        into a pandas DataFrame with columns "Year", "fertilizer_quantity", "output_quantity",
        and "tfp". If these columns are not present, an error will be raised.
        """

        if self.dataset is None:
            self.download_data()

        # Check that the year argument is an int
        if not isinstance(year, int):
            raise TypeError("Year argument must be an integer.")

        # Load the gapminder dataset into a pandas DataFrame
        data_frame = self.dataset

        # Filter the DataFrame to include only the selected year
        data_frame_year = data_frame[data_frame["Year"] == year]

        # Create a scatter plot of fertilizer quantity vs. output quantity
        plt.figure(figsize=(6.4, 4.8))
        axis = sns.scatterplot(
            data=data_frame_year,
            x="fertilizer_quantity",
            y="output_quantity",
            size="tfp",
            sizes=(25, 250),
            alpha=0.6,
        )
        axis.set(
            xlabel="Fertilizer quantity",
            ylabel="Output quantity",
            xscale="log",
            yscale="log",
            title=f"Evolvement of TFP for the year {year}",
        )
        plt.show()
