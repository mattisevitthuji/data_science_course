import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class DataProcessor:
    def __init__(self, df):
        self.df = df

    def clean_data(self):
        self.drop_cols()
        self.fix_na()
        self.create_new_columns()

        return self.df

    def fix_na(self):
        # print na count for each column
        print(self.df.isna().sum(), "\n")
        temp_df = self.df
        # replace all na origin countries with CN because most origin countries are CN
        temp_df['origin_country'] = temp_df['origin_country'].fillna('CN')
        # replace urgency banner, rating_five_count, rating_four_count, rating_three_count,
        # rating_two_count, rating_one_count na with 0
        temp_df['has_urgency_banner'] = temp_df['has_urgency_banner'].fillna(0)
        temp_df['rating_five_count'] = temp_df['rating_five_count'].fillna(0)
        temp_df['rating_four_count'] = temp_df['rating_four_count'].fillna(0)
        temp_df['rating_three_count'] = temp_df['rating_three_count'].fillna(0)
        temp_df['rating_two_count'] = temp_df['rating_two_count'].fillna(0)
        temp_df['rating_one_count'] = temp_df['rating_one_count'].fillna(0)
        # print frequencies of product_color, product_variation_size_id, merchant_info_subtitle
        color_counts = temp_df['product_color'].value_counts().head(10)
        colors = color_counts.index
        freqs = color_counts.values
        # Calculate the number of missing values in the column
        na_count = temp_df['product_color'].isna().sum()
        # Generate random values to fill NA
        fill_values = np.random.choice(colors, size=na_count, p=freqs / freqs.sum())
        # Fill NA values in the column with the generated values
        temp_df.loc[temp_df['product_color'].isna(), 'product_color'] = fill_values
        # replace product_variation_size_id with mose common size: S
        print(temp_df['product_variation_size_id'].value_counts())
        temp_df['product_variation_size_id'] = temp_df['product_variation_size_id'].fillna('S')
        # na urgency_text with "No urgency text"
        temp_df['urgency_text'] = temp_df['urgency_text'].fillna('No urgency text')
        print(temp_df.isna().sum())
        self.df = temp_df

    def create_new_columns(self):
        temp_df = self.df
        temp_df['changed_title'] = np.where(temp_df['title'] == temp_df['title_orig'], 1, 0)


    def visualize_data(self):
        # implement your data visualization here
        # For example, a simple pairplot
        sns.pairplot(self.df)
        plt.show()

    def explore_data(self):
        # implement your data exploration here
        # For example, show basic stats of the dataframe
        print(self.df.describe())

    def drop_cols(self):
        # First we will drop columns that are not useful for our analysis - merchant_profile_picture,
        # merchant_info_subtitle, merchant_id, merchant_name (it is the same for each merchant title)
        # crawl_month, theme, product_picture, product_url, currency_buyer - these are all also irelevant
        temp_df = self.df
        temp_df = temp_df.drop(
            columns=['merchant_profile_picture', 'merchant_info_subtitle', 'merchant_id', 'merchant_name',
                     'crawl_month', 'theme', 'product_picture', 'product_url', 'currency_buyer'])
        self.df = temp_df
