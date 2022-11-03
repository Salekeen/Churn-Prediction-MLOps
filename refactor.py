import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder


def read_csv(filepath):
    df = pd.read_csv(filepath)
    df.drop(['TotalCharges'], inplace=True, axis=1)
    df.set_index("customerID", inplace=True)
    return df


def get_train_test_set(df):
    train_set, test_set = train_test_split(
        df, test_size=0.2, shuffle=True, random_state=42
    )
    return train_set, test_set


def get_X_y(dataset):
    X = dataset.drop("Churn", axis=1)
    y = dataset["Churn"].copy()
    return X,y


def preprocess_data(dataset):
    cat_attrs = ['gender', 'SeniorCitizen', 'Partner', 'Dependents',
                 'PhoneService', 'MultipleLines', 'InternetService',
                 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                 'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                 'PaymentMethod']

    preprocess_pipeline = ColumnTransformer([
        ("cat", OneHotEncoder(), cat_attrs)
    ],
        remainder="passthrough",
        verbose_feature_names_out=False
    )
    dataset_prepared = pd.DataFrame(
        preprocess_pipeline.fit_transform(dataset),
        columns=preprocess_pipeline.get_feature_names_out()
    )
    return dataset_prepared


if __name__ == "__main__":
    datapath = "./data/Telco-Customer-Churn.csv"
    df = read_csv(datapath)
    train_set, test_set = get_train_test_set(df)
    X_train, y_train = get_X_y(train_set)
    X_test, y_test = get_X_y(test_set)
