#Imports and Modules
from sklearn.ensemble import RandomForestClassifier
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

        #Convert string categorical data into numeric data for the RFC to learn
        df = pd.get_dummies(df["Label"])
        df= pd.get_dummies(df["URL"])

    except Exception as e:
        print(f"[Error]:Failed to process data:{e}")
