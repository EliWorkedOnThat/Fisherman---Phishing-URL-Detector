#Imports and Modules
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#Initializing Random Forest Classifier
RFC = RandomForestClassifier(random_state = 0)

#Function to read the CSV Data and process it
def prepare_csv_data():
    try:
        #Read the dataset CSV
        df = pd.read_csv(r"C:\Users\mezin\Desktop\Fisherman\phishing_site_urls.csv")

        #Drop any missing values if any do exist
        df = df.dropna()

        #Create a seperate column for URL length to analyze as a feature
        df["url_length"] = df["URL"].str.len()

        #Create a seperate column to count "/" appearence in a URL
        df["slash_count"] = df["URL"].str.count("/")

        #Create a seperate column to count "." appearence in a URL
        df["dot_count"] = df["URL"].str.count(r"\.")

        #Create a seperate column to count "-" appearence in a URL
        df["hyphen_count"] = df["URL"].str.count("-")

        #Create a seperate column to check the presence of @ in a URL
        df["any_@"] = df["URL"].str.contains("@")

        #Creating an instance of label encoder
        le = LabelEncoder()

        #Returning encoded label from our data
        y = le.fit_transform(df["Label"])

    except Exception as e:
        print(f"[Error]:Failed to process data:{e}")
