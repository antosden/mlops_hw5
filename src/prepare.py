import os
import pandas as pd
import yaml

from sklearn.model_selection import train_test_split


def main():
    with open("params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)

    test_size = params["data"]["test_size"]
    random_state = params["data"]["random_state"]

    os.makedirs("data/processed", exist_ok=True)

    df = pd.read_csv("data/raw/data.csv")
    df.dropna(inplace=True)

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["variety"],
    )

    train_df.to_csv("data/processed/train.csv", index=False)
    test_df.to_csv("data/processed/test.csv", index=False)

    print("Data prepared successfully")


if __name__ == "__main__":
    main()