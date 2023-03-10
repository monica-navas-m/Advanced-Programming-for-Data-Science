## Project Agros Advanced Programming for Data Science - group 10


## Motivation 
In the scope of the course Advanced Programming for Data Science this project allows us to get familiar with distributed version control tools. 

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

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
        

## Installation
The packages required for the Agros class can be found in the yaml file in the git repository. 

## Usage
This project can be used to gain a foundational understanding of agricultural developments worldwide from the year 1961. First, the EDA file contains an exploratory data analysis of the agricultural dataset, including various summary statistics. Secondly, the Agross class as a whole, containing all the methods, can be found in the eponymous python file. Finally, a showcase notebook is created to provide an overview of the methods and findings of this project. 

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
