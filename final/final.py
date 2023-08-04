import pandas as pd

from DataProcessor import DataProcessor

if __name__ == '__main__':
    data = pd.read_csv('summer_products_data.csv')
    data_processor = DataProcessor(data)
    data_processor.clean_data()
    # data.to_pickle('data.pkl')