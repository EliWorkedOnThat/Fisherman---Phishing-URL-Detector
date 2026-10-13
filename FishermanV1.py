import re
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "phishing_site_urls.csv"

SYMBOLS = {
    "question_count": "?",
    "equals_count": "=",
}

EXTENSIONS = {
    "end_html": (".html", ".htm"),
    "end_php": (".php",),
}

FEATURE_COLS = [
    "dot_count",
    "hyphen_count",
    "digit_count",
    "login",
    "domain_length",
    "domain_digit_count",
    "domain_dot_count",
    "domain_hyphen_count",
    "path_length",
    "path_digit_count",
    "path_depth",
    "trailing_slash",
    "has_extension",
] + list(EXTENSIONS) + list(SYMBOLS)


def prepare_csv_data(path=DATA_PATH):
    df = pd.read_csv(path).dropna()

    no_scheme = df["URL"].str.replace(r"^[a-zA-Z]+://", "", regex=True)
    parts = no_scheme.str.split("/", n=1, expand=True)
    domain = parts[0]
    if 1 in parts.columns:
        path = parts[1].fillna("")
    else:
        path = pd.Series("", index=df.index)

    df["dot_count"] = df["URL"].str.count(r"\.")
    df["hyphen_count"] = df["URL"].str.count("-")
    df["digit_count"] = df["URL"].str.count(r"\d")
    df["login"] = df["URL"].str.contains("login", case=False, regex=False).astype(int)

    df["domain_length"] = domain.str.len()
    df["domain_digit_count"] = domain.str.count(r"\d")
    df["domain_dot_count"] = domain.str.count(r"\.")
    df["domain_hyphen_count"] = domain.str.count("-")

    df["path_length"] = path.str.len()
    df["path_digit_count"] = path.str.count(r"\d")
    df["path_depth"] = path.str.count("/")

    df["trailing_slash"] = df["URL"].str.endswith("/").astype(int)
    df["has_extension"] = path.str.contains(r"\.\w{2,5}$").astype(int)

    lower_url = df["URL"].str.lower()
    for col_name, ext in EXTENSIONS.items():
        df[col_name] = lower_url.str.endswith(ext).astype(int)

    for col_name, symbol in SYMBOLS.items():
        df[col_name] = df["URL"].str.count(re.escape(symbol))

    X = df[FEATURE_COLS]
    le = LabelEncoder()
    y = le.fit_transform(df["Label"])

    print(f"Classes: {list(le.classes_)}")
    print(df["Label"].value_counts())

    return X, y


def run_RFC():
    X, y = prepare_csv_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    rfc = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        max_depth=15,
        random_state=42,
        n_jobs=-1,
    )
    rfc.fit(X_train, y_train)

    y_pred = rfc.predict(X_test)

    print(f"Model Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred))
    print(f"Train accuracy: {rfc.score(X_train, y_train):.4f}")

    importances = sorted(zip(X.columns, rfc.feature_importances_), key=lambda p: -p[1])
    for name, score in importances:
        print(f"{name}: {score:.3f}")


if __name__ == "__main__":
    run_RFC()