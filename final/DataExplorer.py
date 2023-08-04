import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

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
        for column in numerical_columns:
            plt.figure()
            sns.boxplot(x=self.df[column])
            title = f"Box plot of {column}"
            plt.title(title, **font_title)
            plt.xlabel(column, **font_label)
            plt.ylabel("Value", **font_label)
            median_value = self.df[column].median()
            plt.text(0.5, 0.5, f"Median: {median_value:.2f}", transform=plt.gca().transAxes,
                     horizontalalignment='center', verticalalignment='center', **font_label)
            plt.savefig("graphs/" + title.replace(" ", "_") + ".png")
            plt.close()

        categorical_columns = ["product_color", "origin_country"]
        for column in categorical_columns:
            plt.figure()
            sns.countplot(data=self.df, x=column)
            title = f"Bar Graph of {column}"
            plt.title(title, **font_title)
            plt.xlabel(column, **font_label)
            plt.ylabel("Count", **font_label)
            value_counts = self.df[column].value_counts()
            for i, count in enumerate(value_counts):
                plt.text(i, count, str(count), ha='center', va='bottom', **font_label)
            plt.savefig("graphs/" + title.replace(" ", "_") + ".png")
            plt.close()

    def correlation_matrix(self):
        # Generate correlation matrix for numeric columns
        corr = self.df.corr()
        sns.heatmap(corr, annot=True)
        plt.title('Correlation Matrix')
        plt.show()
