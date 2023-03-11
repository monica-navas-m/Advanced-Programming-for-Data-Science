## Project Agros Advanced Programming for Data Science - group 10


## Motivation 
In the scope of the course Advanced Programming for Data Science this project allows us to get familiar with distributed version control tools. 

## Description
This project (git@gitlab.com:monica-navas-m/group_10.git) contains class Agros consisting of various methods that perform analyses on agricultural data. The goal of this project is to contribute to the green transition by having a more savvy taskforce, by performing various analyses on countries' agricultural output. One can consult https://www.ers.usda.gov/data-products/international-agricultural-productivity/ to get familiar with the agricultural concepts that are central in this project. 

## Sources
This project depends on the "Agricultuar Total Factor Productivity (USDA)" dataset that can be found here: https://github.com/owid/owid-datasets/blob/master/datasets/Agricultural%20total%20factor%20productivity%20(USDA)/Agricultural%20total%20factor%20productivity%20(USDA).csv 

## Repository
Exploratory data analyses of the agricultural dataset can be found in EDA.ipynb. The class containing all methods can be found in the agros.py file in the Class folder. In addition, crucial files for housekeeping include: changelog, LICENSE and gitignore. Finally, a summary of the analyses and conclusions can be found in the showcase notebook. 

## Installation 
Ideally, one can create a virtual environment dedicated to running this script from the environment.yaml file that can be found in this repository. Alternatively, one should make sure the following libraries are installed in order for this script to run: 

import os
import urllib.request
from typing import List, Optional, Union
import pandas as pd
import seaborn as sns
import geopandas as gpd
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from statsmodels.tsa.arima.model import ARIMA

## How to run?
You can run desired functions by importing the agros class in a jupyter notebook (best option for graphical visualizations).

For example: to generate a list with unique countries (using the get_countries method):

from agros import Agros

example = Agros()

print(example.get_countries())


## Code Examples
The method below is an example of a method included in the Agros class. It returns the list of all unique countries in the agriculture dataframe.

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
       
## Methods 
The Agross class consists of nine methods in total. The first two methods are dedicated to downloading the required dataset and making it feasible for further analyses (by saving it into a pandas dataframe). The "get_countries" method returns a list with unique countries in the dataset. The "plot_correlation" method generates a heatmap, plotting the  correlation between the various "\_quantity" columns. The "areachart_country_output" method plots an area chart of the distinct "\_output_" columns. The "plot_country_output" method compares the total of the "\_output_" columns for each of the chosen countries and plot it, so a comparison can be made. The "gapminder method" plots a scatter plot where x is fertilizer_quantity, y is output_quantity, and the area of each dot represents the tfp (total factor productivity). The Choropleth method plots the tfp on the worldpad, using geopandas to create a choropleth map. Finally, the "predictory method plots the tfp, and uses an ARIMA to predict it up to 2050.

## Authors and acknowledgment
The authors of this project include: Laura Weil, Sebastian Varadappa, Monica Navas, Gabriel Abib.

## Contact information
In case of further questions, feel free to contact the authors at: 
53012@novasbe.pt (Laura Weil)
55435@novasbe.pt (Sebastian Varadappa)
54577@novasbe.pt (Monica Navas)
55646@novasbe.pt (Gabriel Abib) 

## License
Licensed under the Apache License, Version 2.0 (the "License").

## Project status
This project is submitted for grading.
