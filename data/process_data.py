import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
        Doc String 
        Description: This function extracts data from the csv files and creates a dataframe with all the data  
        Input: path to the messages.csv and categories.csv
        Output: dataframe with merged messages and categories
    '''
    # load messages dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on='id')
    return df

def clean_data(df):
    
    '''
        Doc String 
        Description: This function cleans the dataframe to organises the data 
        Input: dataframe - from the load_data function
        Output: dataframe - with cleaned data
    '''
    
    categories = df.categories.str.split(';', expand=True)
    row = categories.iloc[0]
    category_colnames = [category[:-2] for category in categories.iloc[0]]
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str.split('-', expand=True)[1]

        # convert column from string to numeric
        categories[column] = categories[column].astype('int')
    df = df.drop('categories', axis = 1)
    df = pd.merge(df,categories, left_index=True, right_index=True)
    # check number of duplicates
    for column in ['id', 'message','original', 'genre']:
        print(column, len(df[column]) - len(df[column].drop_duplicates()))
    df = df.drop_duplicates(subset=['id', 'message','original', 'genre'])
    return df


def save_data(df, database_filename):
    
    '''
        Doc String 
        Description: This function saves the data to the database 
        Input: dataframe, path to the database
        Output: None
    '''
    
    engine = create_engine("sqlite:///"+database_filename)
    df.to_sql('disaster_table', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()