import pandas as pd
from PIL import Image
import os
import pickle

def read_csv(filename):
    """
    Read CSV file from 'data' folder
    
    Args:
        filename (str): Name of the CSV file
    
    Returns:
        pandas.DataFrame: Dataframe containing CSV data
    """
    path = os.path.join('data', filename)

    if os.path.exists(path):
        return pd.read_csv(path)

    raise FileNotFoundError(f"Could not find {filename} in data directory")


def read_image(filename):
    """
    Read image file from 'streamlit_plots_and_figures' folder
    
    Args:
        filename (str): Name of the image file
    
    Returns:
        PIL.Image: Image object
    """

    path = os.path.join('streamlit_plots_and_figures', filename)


    if os.path.exists(path):
        return Image.open(path)
    
    raise FileNotFoundError(f"Could not find {filename} in streamlit_plots_and_figures directory")


def read_model(filename):
    """
    Read pickle file from 'saved' subfolder in models

    Args:
        filename (str): Name of the pickle file

    Returns:
        pickle object
    """
    path = os.path.join('models/saved', filename)
    if os.path.exists(path):
        with open(path, 'rb') as file:
            return pickle.load(file)

    raise FileNotFoundError(f"Could not find {filename} in models/saved directory")


