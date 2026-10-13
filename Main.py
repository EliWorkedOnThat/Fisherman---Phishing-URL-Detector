#Imports and Modules
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

#Function to read the CSV Data and process it
def prepare_csv_data():
    try:
        #Read the dataset CSV
        df = pd.read_csv(r"C:\Users\mezin\Desktop\Fisherman\phishing_site_urls.csv")

        #Drop any missing values if any do exist
        df = df.dropna()

        #Symbols to count, mapped to safe column names
        symbols = {
            "question_count": "?",
            "equals_count": "=",
            "ampersand_count": "&",
            "percent_count": "%",
        }

        #Feature columns built from the URL text
        df["url_length"] = df["URL"].str.len()
        df["slash_count"] = df["URL"].str.count("/")
        df["dot_count"] = df["URL"].str.count(r"\.")
        df["hyphen_count"] = df["URL"].str.count("-")
        df["login"] = df["URL"].str.contains("login").astype(int)
        df["digit_count"] = df["URL"].str.count(r"\d")

        #One count column per symbol (re.escape handles regex characters like ?)
        for col_name, symbol in symbols.items():
            df[col_name] = df["URL"].str.count(re.escape(symbol))

        #Feature list built once, so new columns can't be forgotten
        feature_cols = [
            "url_length", "slash_count", "dot_count", "hyphen_count",
            "login", "digit_count",
        ] + list(symbols.keys())

        X = df[feature_cols]

        #Encode the label (target)
        le = LabelEncoder()
        y = le.fit_transform(df["Label"])

        print(f"Classes: {list(le.classes_)}")
        print(df["Label"].value_counts())

        return X, y

    except Exception as e:
        print(f"[Error]:Failed to process data:{e}")
        raise  # re-raise so the real error shows instead of returning None

#Function to run the RFC across several depths
def run_RFC():

    #Load the prepared data
    X, y = prepare_csv_data()

    #Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    #Depths to compare (None = no limit)
    depths = [8, 10, 12, 15, 20, 25, None]

    print(f"\n{'depth':>6} {'test':>7} {'train':>7} {'gap':>7} {'bad_rec':>8} {'bad_prec':>9}")

    for depth in depths:
        #Only max_depth changes between runs
        rfc = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            max_depth=depth,
            random_state=42,
            n_jobs=-1,
        )
        rfc.fit(X_train, y_train)

        y_pred = rfc.predict(X_test)

        test_acc = accuracy_score(y_test, y_pred)
        train_acc = rfc.score(X_train, y_train)

        #Class 0 is "bad" (LabelEncoder sorts alphabetically)
        bad_recall = recall_score(y_test, y_pred, pos_label=0)
        bad_precision = precision_score(y_test, y_pred, pos_label=0)

        print(f"{str(depth):>6} {test_acc:>7.4f} {train_acc:>7.4f} "
              f"{train_acc - test_acc:>7.4f} {bad_recall:>8.2f} {bad_precision:>9.2f}")

if __name__ == "__main__":
    run_RFC()