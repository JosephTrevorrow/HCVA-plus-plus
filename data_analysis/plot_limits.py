import pandas as pd
import matplotlib.pyplot as plt
import scienceplots
import numpy as np
import seaborn as sns
import os


def plot_data(data: pd.DataFrame, title: str):
    """
    This function plots the limit P data (-t True).
    INPUT: data -- pd.DataFrame, title -- str (title of the plot)
    """
    plt.figure(figsize=(6, 3))
    # Set the style of the plots
    plt.style.use("ggplot")
    plt.grid(True)

    plt.plot(data["p"], data["Dist_p"], label="$||P^{(1)}_S-P^{(P)}_S||_p$")
    plt.plot(data["p"], data["Dist_inf"], label="$||P^{(\infty)}_S-P^{(P)}_S||_p$")

    # Scale the x-axis
    plt.xlim(1, 3)
    plt.fill_between(data["p"], data["Dist_p"], data["Dist_inf"],
                     where=(data["p"] >= 1.1) & (data["p"] <= 2.3),
                     color='green', alpha=0.3)
    plt.fill_between(data["p"], data["Dist_p"], data["Dist_inf"],
                     where=(data["p"] >= 2.2) & (data["p"] <= 5),
                     color='blue', alpha=0.3)

    plt.xlabel("p")
    plt.ylabel("Distance")
    plt.legend(loc='upper right')
    plt.savefig("plot_data.png")


data = pd.read_csv("/Users/josephtrevorrow/Documents/GitHub/HCVA-plus-plus/results/placeholder_results/CASE1_Limits.csv")
plot_data(data, "Value Principle Preference Matrix Aggregation (limit P)")
plt.show()
