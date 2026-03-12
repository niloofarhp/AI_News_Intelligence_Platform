import pandas as pd


def preprocess(df):

    df = df[["headline", "short_description", "category", "date", "link"]]

    df = df.dropna()

    df["full_text"] = df["headline"] + ". " + df["short_description"]

    df = df.drop_duplicates(subset=["full_text"])

    df = df[df["full_text"].str.len() > 50]

    return df


if __name__ == "__main__":

    input_path = "data/raw/News_Category_Dataset_v3.json"
    output_path = "data/processed/cleaned_articles.csv"

    df = pd.read_json(input_path, lines=True)

    df_clean = preprocess(df)

    print("Cleaned dataset size:", df_clean.shape)

    df_clean.to_csv(output_path, index=False)

    print("Saved cleaned dataset.")