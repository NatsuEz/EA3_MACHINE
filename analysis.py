
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def main():
    df=pd.read_csv("synthetic_training_data.csv")
    features=['income','age','loan_amount','tenure_months','late_payments']
    X=df[features]
    X_scaled=StandardScaler().fit_transform(X)
    km=KMeans(n_clusters=4,random_state=0).fit(X_scaled)
    df['cluster']=km.labels_
    print(df.head())

if __name__=="__main__":
    main()
