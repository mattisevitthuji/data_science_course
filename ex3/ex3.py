import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', None)


class DataSet:
    def __init__(self, df):
        self.df = df

    def compute_statistical_summary(self):
        # print missing values na in df
        print(self.df.isna().sum(), "\n")
        # print if any customer ids are duplicated
        print("Number of duplicated customer ids: ", self.df['Customer_Id'].duplicated().sum(), "\n")
        numerical_columns = self.df.select_dtypes(include=[np.number])
        if not numerical_columns.empty:
            numerical_stats = numerical_columns.agg(['mean', 'median', 'std'])
            print("Statistics for numerical columns:")
            print(numerical_stats)
        else:
            print("No numerical columns found.")

        categorical_columns = self.df.select_dtypes(include=['object', 'category'])
        if not categorical_columns.empty:
            print("\nFrequency distribution for categorical columns:")
            for column in categorical_columns:
                freq_dist = ((self.df[column].value_counts() / len(df)) * 100).astype(str) + ' %'
                print(f"\nColumn: {column}")
                print(freq_dist)

    def visualize_data(self):
        sns.set_palette('bright')
        font_title = {'fontweight': 'bold', 'fontsize': 14}
        font_label = {'fontweight': 'bold', 'fontsize': 12}
        numerical_columns = self.df.select_dtypes(include=[np.number])
        for column in numerical_columns:
            plt.figure()
            sns.histplot(self.df[column])
            plt.title(f"Histogram of {column}", **font_title)
            plt.xlabel(column, **font_label)
            plt.ylabel("Frequency", **font_label)
            plt.show()
            plt.show()
        for column in numerical_columns:
            plt.figure()
            sns.boxplot(x=self.df[column])
            plt.title(f"Box plot of {column}", **font_title)
            plt.xlabel(column, **font_label)
            plt.ylabel("Value", **font_label)
            median_value = df[column].median()
            plt.text(0.5, 0.5, f"Median: {median_value:.2f}", transform=plt.gca().transAxes,
                     horizontalalignment='center', verticalalignment='center', **font_label)
            plt.show()
        sns.set(font_scale=1.2)
        sns.pairplot(self.df[numerical_columns.columns])
        plt.title("Scatter Plot Matrix", **font_title)
        plt.show()
        categorical_columns = df.select_dtypes(include=['object', 'category'])
        for column in categorical_columns:
            plt.figure()
            sns.countplot(data=df, x=column)
            plt.title(f"Bar Graph of {column}", **font_title)
            plt.xlabel(column, **font_label)
            plt.ylabel("Count", **font_label)
            value_counts = df[column].value_counts()
            for i, count in enumerate(value_counts):
                plt.text(i, count, str(count), ha='center', va='bottom', **font_label)
            plt.show()

    def preprocess_data(self):
        numerical_columns = self.df.select_dtypes(include=[np.number])
        scaler = MinMaxScaler()
        self.df[numerical_columns.columns] = scaler.fit_transform(self.df[numerical_columns.columns])
        categorical_columns = self.df.select_dtypes(include=['object', 'category']).columns.tolist()
        self.df = pd.get_dummies(self.df, columns=categorical_columns)

    def find_optimal_clusters(self):
        X = self.df.drop('Customer_Id', axis=1)
        silhouette_scores = []
        k_range = range(2, 21)
        max_k = 2
        max_score = -1
        for k in k_range:
            print(f"Running for k = {k}")
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(X)
            labels = kmeans.predict(X)
            score = silhouette_score(X, labels)
            silhouette_scores.append(score)
            if score > max_score:
                max_k = k
                max_score = score
        print("Silhouette scores: ", silhouette_scores)
        return max_k

    def perform_cluster_and_analyze(self, optimal_clusters):
        X = self.df.drop('Customer_Id', axis=1)
        kmeans = KMeans(n_clusters=optimal_clusters)
        kmeans.fit(X)
        print("Number of customers in each cluster:", kmeans.labels_.size)
        cluster_labels = kmeans.labels_
        unique_labels, counts = np.unique(cluster_labels, return_counts=True)
        for cluster_label, count in zip(unique_labels, counts):
            print(f"Cluster {cluster_label}: {count} samples")
        X['Cluster'] = cluster_labels
        cluster_means = X.groupby('Cluster').mean()
        print(cluster_means)
        cluster_means.to_csv('cluster_means.csv')


if __name__ == '__main__':
    df = pd.read_csv("JustBuy_data.csv")
    data_set = DataSet(df)
    data_set.compute_statistical_summary()
    # data_set.visualize_data()
    # data_set.preprocess_data()
    # optimal_clusters = data_set.find_optimal_clusters()
    # print("Optimal number of clusters: ", optimal_clusters)
    # data_set.perform_cluster_and_analyze(optimal_clusters)

