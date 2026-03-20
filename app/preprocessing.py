import pandas as pd

def encode_chest_pain_type ( df ):
    df['chest_pain_type'] = df['chest_pain_type'].map({

        'asymptomatic' : 0, 'atypical angina': 1,
        'non-anginal pain': 2, 'typical angina': 3
    })
    return df



def encode_thalassemia ( df ):
    df["thalassemia"] = df["thalassemia"].map({
        'normal': 0, 'fixed defect': 1,
        'reversable defect': 2
    })
    return df
   
# ACHTUNG: 'diagnosis' muss aus der Liste apply_1_hot_encoding in preprocessing.py entfernt werden, 


def apply_1_hot_encoding(df, full_df):
    cols = ['sex', 'fasting_blood_sugar', 'rest_ecg', 'st_slope', 'exercise_induced_angina']
    
    # 1. Kategorien fixieren (damit Spaltenanzahl immer gleich bleibt)
    for col in cols:
        df[col] = pd.Categorical(df[col], categories=full_df[col].unique())
    
    # 2. Umwandeln (dtype=int macht direkt 0/1 statt True/False)
    return pd.get_dummies(df, columns=cols, dtype=int)


