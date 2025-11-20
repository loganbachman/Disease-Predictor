import pandas as pd
import matplotlib.pyplot as plt
import os
from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore
from sklearn.model_selection import train_test_split

plt.style.use('ggplot')


def load_dataset(data_path='data/raw', use_full_data=True):
    # Create path for data
    os.makedirs(data_path, exist_ok=True)
    csv_name = 'Final_Augmented_dataset_Diseases_and_Symptoms.csv'
    csv_path = os.path.join(data_path, csv_name)
    # if data doesn't exist, call Kaggle Api to load
    if not os.path.exists(csv_path):
        print("Downloading kaggle dataset...")
        api = KaggleApi()
        api.authenticate()
        api.dataset_download_files(
            'dhivyeshrk/diseases-and-symptoms-dataset',
            path=data_path,
            unzip=True
        )

    dataset = pd.read_csv(csv_path)
    dataset.dropna(inplace=True)
    # X (features) is all columns (symptoms) except first (diseases)
    X = dataset.drop('diseases', axis=1)
    # y (target) is the resulting disease, what we are trying to predict
    y = dataset['diseases']

    # filter out diseases that dont appear at least 10 times
    disease_counts = y.value_counts()
    valid_diseases = disease_counts[disease_counts >= 10].index
    valid_mask = y.isin(valid_diseases)
    X = X[valid_mask]
    y = y[valid_mask]

    # if NOT using full dataset
    if not use_full_data:
        # Take a stratified sample of 50K from dataset
        X, _, y, _ = train_test_split(
            X, y,
            train_size=50000, 
            stratify=y, 
            random_state=42
        )

    return X, y


def main():
    X, y = load_dataset(use_full_data=True)

    diseases_counts = y.value_counts()

    plt.figure(figsize=(14, 6))
    ax = diseases_counts.head(10).plot(
        kind="bar",
        title="Top 25 Diseases in Dataset",
        xlabel="Disease",
        ylabel="Number of Records"
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha="center")
    plt.show()
    top_diseases = diseases_counts.head(25)
    print(top_diseases)


if __name__ == "__main__":
    main()
