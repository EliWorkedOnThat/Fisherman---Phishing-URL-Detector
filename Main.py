#Imports and Modules
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

        #Feature columns built from the URL text
        df["url_length"] = df["URL"].str.len()
        df["slash_count"] = df["URL"].str.count("/")
        df["dot_count"] = df["URL"].str.count(r"\.")
        df["hyphen_count"] = df["URL"].str.count("-")
        df["has_at"] = df["URL"].str.contains("@", regex=False).astype(int)
        df["login"] = df["URL"].str.contains("login").astype(int)
        df["secure"] = df["URL"].str.contains("secure").astype(int)
        df["digit_count"] = df["URL"].str.count(r"\d")
        df["http_vs_https"] = df["URL"].str.startswith("https")

        #Data frame of the features the model will learn from
        X = df[["url_length", "slash_count", "dot_count", "hyphen_count", "has_at" , "login" , "secure" , "digit_count" , "http_vs_https"]]

        #Encode the label (target)
        le = LabelEncoder()
        y = le.fit_transform(df["Label"])

        print(f"Classes: {list(le.classes_)}")
        print(df["Label"].value_counts())

        return X, y

    except Exception as e:
        print(f"[Error]:Failed to process data:{e}")
        raise  # re-raise so the real error shows instead of returning None

#Function to run the RFC
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

    #Initialize and train the model
    rfc = RandomForestClassifier(n_estimators=300, class_weight="balanced" ,random_state=42)
    rfc.fit(X_train, y_train)

    #Predict on the unseen test data
    y_pred = rfc.predict(X_test)

    #Evaluate
    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    print(f"Train accuracy: {rfc.score(X_train, y_train):.4f}")

    #See which features the model relied on
    for name, score in zip(X.columns, rfc.feature_importances_):
        print(f"{name}: {score:.3f}")

if __name__ == "__main__":
    run_RFC()