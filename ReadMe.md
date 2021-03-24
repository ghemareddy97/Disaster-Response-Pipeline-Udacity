# Disaster Response Pipeline Project

## Contents

* Instructions
* Installation
* Motivation
* File Structure
* Results
* Licensing and Acknowledgments


### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.(make sure to run train_classifier.py to generate the pkl file)
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Installation

Basic [Anaconda](https://www.anaconda.com/) installation would be enough to run this code. The following packages have been used to create this notebook.

| Library | Usage |
| ----------------- | ----------- |
|Pandas|to handle and manipulate data|
| Numpy | to perform numpy array operations |
| Matplotlib | for data visualization |
| re | regular expressions for performing string operations on the data in the dataframe |
| flask | Webserver for serving backend  |

### Motivation

This project is desgined and built to categorize the disaster into one or more of the available 36 different categories for ease of response. A machine learning model with MultiOutputClassifier and RandomForestClassifier have been trained on a variety of data with a large number of samples for each category. 

### File Structure

- app
    - templates
        - go.html
        - master.html
    - run.py
- data
    - disasterResponse.db
    - disaster_categories.csv
    - disaster_messages.csv
    - process_data.py
- models
    - classifier.pkl
    - train_classifier.py
- README.md

### Results

Please run train_classifier.py to get the classification report on the test data for each category. The documentation and instructions for running each python file are available above in the instructions section and also in the file itself.

## Licensing and Acknowledgements
This code is part of the Udacity project for data science nanodegree. I don't hold rights for the data. Feel free to use the code. 