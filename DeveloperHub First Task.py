

# Exploring and visualizing a Dataset: Iris Dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df=pd.read_csv("C:/Users/User/Downloads/iris.csv")
print(df.head())

print(df.shape)

print(df.columns)

print(df.describe())

print(df.info())

sns.scatterplot(x="SepalLengthCm", y="SepalWidthCm", data=df,hue="Species")
plt.title("Sepal Length vs Sepal Width")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Sepal Width (cm)")
plt.legend(title="Species")

plt.show()


sns.histplot(df["SepalLengthCm"], kde=True, bins=10, color="LightBlue")
plt.title("Distribution of Sepal Length with KDE")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Frequency")
plt.show()

sns.boxplot(x="SepalLengthCm",y="Species", data=df)
plt.title("Box Plot of Sepal Length")
plt.xlabel("Sepal Length (cm)")
plt.ylabel("Species")
plt.show()
