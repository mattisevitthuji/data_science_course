import pandas as pd

from DataProcessor import DataProcessor
from DataExplorer import DataExplorer

if __name__ == '__main__':
    data = pd.read_csv('summer_products_data.csv')
    data_processor = DataProcessor(data)
    data_processor.clean_data()
    DataExplorer(data_processor.df).basic_stats()
    DataExplorer(data_processor.df).visualize_data()
    # data.to_pickle('data.pkl')