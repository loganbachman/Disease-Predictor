import pandas as pd
import matplotlib.pyplot as plt
import os
from kaggle.api.kaggle_api_extended import KaggleApi #type: ignore

plt.style.use('ggplot')

api = KaggleApi()
api.authenticate()
def main():
    api.dataset_download_files('dhivyeshrk/diseases-and-symptoms-dataset', unzip=True)
    csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    dataset = pd.read_csv('Final_Augmented_dataset_Diseases_and_Symptoms.csv')
    # X (features) is all columns (symptoms) except first (diseases)
    X = dataset.drop('diseases', axis=1)
    # y (target) is the resulting disease, what we are trying to predict
    y = dataset['diseases']
    
    #Display general dataset info
    print(f"\nDataset shape: {dataset.shape}")
    print(f"\nColumns: {dataset.columns.tolist()}")
    print(f"\nFirst few rows:\n{dataset.head()}")
    
    diseases_counts = y.value_counts()
    
    plt.figure(figsize=(14, 6))
    ax = diseases_counts.head(25).plot(
        kind="bar",
        title="Top 25 Diseases in Dataset",
        xlabel="Disease",
        ylabel="Number of Records"
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
    plt.show()
    

if __name__ == "__main__":
    main()
