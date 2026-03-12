import pandas as pd
import json


def load_dataset(path):
    records = []

    with open(path, "r") as f:
        for line in f:
            records.append(json.loads(line))

    df = pd.DataFrame(records)

    return df


if __name__ == "__main__":
    dataset_path = "data/raw/News_Category_Dataset_v3.json"

    df = load_dataset(dataset_path)

    print("Dataset shape:", df.shape)
    print(df.head())