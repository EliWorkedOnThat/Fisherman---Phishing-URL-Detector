#Imports and Modules
import re
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

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
        df["session"] = df["URL"].str.contains("session")
        df["apk"] = df["URL"].str.contains(r"\.apk").astype(int)
        df["zip"] =  df["URL"].str.contains(r"\.zip").astype(int)
        df["end_html"] = df["URL"].str.endswith((".html", ".htm")).astype(int)

        #One count column per symbol (re.escape handles regex characters like ?)
        for col_name, symbol in symbols.items():
            df[col_name] = df["URL"].str.count(re.escape(symbol))

        #Feature list built once, so new columns can't be forgotten
        feature_cols = [
            "url_length", "slash_count","apk","end_html" ,"zip", "dot_count","session", "hyphen_count",
            "login", "digit_count",
        ] + list(symbols.keys())

        #Data frame of the features the model will learn from
        X = df[feature_cols]

        #Encode the label (target): class 0 = "bad", class 1 = "good"
        le = LabelEncoder()
        y = le.fit_transform(df["Label"])

        #Show the class names and how balanced the data is
        print(f"Classes: {list(le.classes_)}")
        print(df["Label"].value_counts())

        return X, y

    except Exception as e:
        print(f"[Error]:Failed to process data:{e}")
        raise  # re-raise so the real error shows instead of returning None

#Function to train and evaluate the Random Forest
def run_RFC():

    #Load the prepared features and labels
    X, y = prepare_csv_data()

    #Split into training and testing sets (stratify keeps the class ratio equal in both)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    #Initialize and train the model
    #max_depth=15 gave the best test accuracy in the depth comparison
    #class_weight="balanced" compensates for there being more good URLs than bad ones
    #n_jobs=-1 uses all CPU cores to speed up training
    rfc = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        max_depth=15,
        random_state=42,
        n_jobs=-1,
    )
    rfc.fit(X_train, y_train)

    #Predict on the unseen test data
    y_pred = rfc.predict(X_test)

    #Evaluate: accuracy plus per-class precision and recall
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))

    #Train accuracy, to compare against test accuracy and spot overfitting
    print(f"Train accuracy: {rfc.score(X_train, y_train):.4f}")

    #See which features the model relied on
    for name, score in zip(X.columns, rfc.feature_importances_):
        print(f"{name}: {score:.3f}")

#Only run when the script is executed directly, not when imported
if __name__ == "__main__":
    run_RFC()