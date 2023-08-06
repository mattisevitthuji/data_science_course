import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import geopandas as gpd

class DataExplorer:
    def __init__(self, df):
        self.df = df

    def basic_stats(self):
        # Print basic statistics for each column
        pd.set_option('display.max_columns', None)
        numerical_columns = self.df.select_dtypes(include=[np.number])
        stats = numerical_columns.agg(['mean', 'median', 'std'])
        stats.to_csv('stats.csv')

    def visualize_data(self):
        sns.set_palette('bright')
        font_title = {'fontweight': 'bold', 'fontsize': 14}
        font_label = {'fontweight': 'bold', 'fontsize': 12}
        numerical_columns = self.df.select_dtypes(include=[np.number])
        cols_to_drop = [col for col in numerical_columns.columns if 'product_color_' in col or '_tag' in col]

        # Drop these columns from the DataFrame
        numerical_columns = numerical_columns.drop(columns=cols_to_drop)
        numerical_columns = numerical_columns
        for column in numerical_columns:
            plt.figure()
            sns.histplot(self.df[column])
            title = f"Histogram of {column}"
            plt.title(title, **font_title)
            plt.xlabel(column, **font_label)
            plt.ylabel("Frequency", **font_label)
            plt.savefig("graphs/" + title.replace(" ", "_") + ".png")
            plt.close()

        # origin_country graph
        column = "origin_country"
        plt.figure()
        sns.countplot(data=self.df, x=column)
        title = f"Bar Graph of {column}"
        plt.title(title, **font_title)
        plt.xlabel(column, **font_label)
        plt.ylabel("Count", **font_label)
        plt.savefig("graphs/" + title.replace(" ", "_") + ".png")
        plt.close()

        # color graph
        color_counts = self.df['product_color'].value_counts()
        color_mapping = {
            'other': 'grey'
        }
        bar_colors = {"multicolor": 'red'}
        hatch_patterns = {"multicolor": '///'}
        hatch_colors = {"multicolor": 'blue'}
        colors = color_counts.index.tolist()
        counts = color_counts.values.tolist()
        fig, ax = plt.subplots()
        for i, color in enumerate(colors):
            if color in hatch_patterns:
                ax.bar(i, counts[i], color=bar_colors[color], edgecolor=hatch_colors[color],
                       hatch=hatch_patterns[color])
                continue
            ax.bar(i, counts[i], edgecolor='black', color=color_mapping.get(color, color))
        title = f"Bar Graph of product_color"
        plt.title(title, **font_title)
        ax.set_xticks(range(len(colors)))
        plt.xticks(rotation=30, size=8)
        ax.set_xticklabels(colors)
        plt.savefig("graphs/" + title.replace(" ", "_") + ".png")
        plt.close()

        # price vs retail price
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.df, x='price', y='retail_price')
        title = 'Price vs Retail Price'
        plt.title(title)
        plt.savefig("graphs/" + title.replace(" ", "_") + ".png")
        plt.close()

        # Visualize the relationship between rating and units_sold with a scatter plot
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.df, x='rating', y='units_sold')
        title = 'Rating vs Units Sold'
        plt.title(title)
        plt.savefig("graphs/" + title.replace(" ", "_") + ".png")
        plt.close()

        # Compare merchant_rating with rating
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.df, x='merchant_rating', y='rating')
        title = 'Merchant Rating vs Rating'
        plt.title(title)
        plt.savefig("graphs/" + title.replace(" ", "_") + ".png")
        plt.close()

        # Stacked bar plot for breakdowns of rating columns
        ratings = ['rating_five_count', 'rating_four_count', 'rating_three_count', 'rating_two_count',
                   'rating_one_count']
        rating_sums = self.df[ratings].sum()
        star_labels = ['★' * i for i in range(5, 0, -1)]
        fig, ax = plt.subplots(figsize=(10, 6))
        rating_sums.plot(kind='bar', ax=ax)
        ax.set_xticklabels(star_labels, rotation=0)
        title = 'Breakdown of Ratings'
        plt.title(title)
        plt.ylabel('Count')
        plt.xlabel('Rating')
        plt.tight_layout()
        plt.savefig("graphs/" + title.replace(" ", "_") + ".png")
        plt.close()

        # Correlation heatmap for rating against other numeric columns
        correlation_matrix = self.df[
            ['rating', 'price', 'retail_price', 'units_sold', 'merchant_rating_count', 'merchant_rating']].corr()
        plt.figure(figsize=(10, 6))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
        title = 'Correlation Heatmap'
        plt.title(title)
        # make ticks bold
        plt.yticks(rotation=20, size=7, weight='bold')
        plt.xticks(rotation=10, size=8, weight='bold')
        plt.savefig("graphs/" + title.replace(" ", "_") + ".png")
        plt.close()

        #scatter plot for price vs units_sold
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.df, x='price', y='units_sold')
        title = 'Price vs Units Sold'
        plt.title(title)
        plt.savefig("graphs/" + title.replace(" ", "_") + ".png")
        plt.close()

        # map graph
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        df_country_count = self.df['origin_country'].value_counts().reset_index()
        df_country_count.columns = ['country', 'count']

        merged = world.set_index('iso_a3').join(df_country_count.set_index('country'))
        merged['count'] = merged['count'].fillna(0)
        fig, ax = plt.subplots(1, 1, figsize=(15, 25))
        merged.plot(column='count', cmap='OrRd', linewidth=0.8, ax=ax, edgecolor='0.8', legend=True)
        title = 'Products Origin by Country'
        ax.set_title(title)
        plt.savefig("graphs/" + title.replace(" ", "_") + ".png")
        plt.close()
