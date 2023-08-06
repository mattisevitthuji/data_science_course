import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


class DataProcessor:
    def __init__(self, df):
        self.df = df

    def clean_data(self):
        self.fix_na()
        self.create_new_columns()

        self.drop_cols()
        return self.df

    def fix_na(self):
        print(self.df.columns)
        # print columns with na
        print(self.df.columns[self.df.isna().any()].tolist(), "\n")

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

        # find most frequent colors and replace na with the according to frequency
        color_counts = temp_df['product_color'].value_counts().head(10)
        colors = color_counts.index
        freqs = color_counts.values
        # Calculate the number of missing values in the column
        na_count = temp_df['product_color'].isna().sum()
        # Generate random values to fill NA
        fill_values = np.random.choice(colors, size=na_count, p=freqs / freqs.sum())
        # Fill NA values in the column with the generated values
        temp_df.loc[temp_df['product_color'].isna(), 'product_color'] = fill_values
        self.df = temp_df

    def create_new_columns(self):
        temp_df = self.df
        # new feature that check if item title was changed
        temp_df['changed_title'] = np.where(temp_df['title'] == temp_df['title_orig'], 1, 0)
        temp_df = temp_df.drop(columns=['title_orig'])

        # new feature that keep all the tags that appear more than 50 times as on hot encoded column
        tags = temp_df['tags'].str.split(',', expand=True).stack().value_counts()
        tags = tags[tags > 100]
        tags = tags.index
        for tag in tags:
            temp_df[tag + "_tag"] = temp_df['tags'].str.contains(tag).astype(int)

        # new feature that holds tag count
        temp_df['tag_count'] = temp_df['tags'].str.split(',').str.len()

        # drop tags
        temp_df = temp_df.drop(columns=['tags'])

        # new feature that shows difference between actual price and retail price
        temp_df['price_diff'] = temp_df['price'] - temp_df['retail_price']

        # new feature that one hot encodes the most common colors
        temp_df = self.fix_color_col(temp_df)
        one_hot = pd.get_dummies(temp_df['product_color']).astype(int)
        # temp_df = temp_df.drop(columns=['product_color'])
        temp_df = pd.concat([temp_df, one_hot], axis=1)

        # new feature that one hot encodes if shipping_option_name is Livraison standard or not
        temp_df['is_livraison_standard_shipping'] = np.where(temp_df['shipping_option_name'] == 'Livraison standard', 1,0)
        temp_df = temp_df.drop(columns=['shipping_option_name'])

        # new feature that one hot encodes if origin_country is CN or not
        temp_df['origin_country'].replace("CN", "CHN", inplace=True)
        temp_df['origin_country'].replace("US", "UAS", inplace=True)
        temp_df['origin_country'].replace("VE", "VEN", inplace=True)
        temp_df['origin_country'].replace("GB", "GBR", inplace=True)
        temp_df['origin_country'].replace("AT", "AUT", inplace=True)
        one_hot = pd.get_dummies(temp_df['origin_country']).astype(int)
        # temp_df = temp_df.drop(columns=['origin_country'])
        temp_df = pd.concat([temp_df, one_hot], axis=1)
        self.df = temp_df


    def fix_color_col(self, temp_df):
        temp_df['product_color'] = temp_df['product_color'].str.lower()
        temp_df['product_color'].replace("grey", "gray", inplace=True)
        top_colors = temp_df['product_color'].value_counts().index[:18].tolist()

        def match_colors(color_option):
            for color in top_colors:
                if color in color_option: return color
            return "other"

        temp_df['product_color'] = temp_df['product_color'].apply(match_colors)
        return temp_df

    def drop_cols(self):
        # First we will drop columns that are not useful for our analysis - merchant_profile_picture,
        # merchant_info_subtitle, merchant_id, merchant_name (it is the same for each merchant title)
        # crawl_month, theme, product_picture, product_url, currency_buyer - these are all also irelevant
        temp_df = self.df
        temp_df = temp_df.drop(
            columns=['merchant_profile_picture', 'merchant_info_subtitle', 'merchant_id', 'merchant_name',
                     'crawl_month', 'theme', 'product_picture', 'product_url', 'currency_buyer',
                     'product_variation_size_id', 'product_id', 'merchant_title', 'urgency_text'])
        self.df = temp_df
