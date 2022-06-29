import numpy as np
import pandas as pd
import seaborn as sn
cars = pd.read_csv('cars_dataset.csv')
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import warnings
warnings.simplefilter(action='ignore', category=Warning)

class LogisticR():

    def __init__(self,data):
        self.data = cars[cars["Make"] == data]
    def feature_engineering(self):
        self.litre = self.data.insert(2, 'litre', pd.Series(self.data["mileage"]*(self.data["mpg"]*0.43))/100)
        self.litre = self.data["litre"]
        return self.litre
    def dummy(self):
        self.data =pd.get_dummies(self.data,columns=['transmission','fuelType'],drop_first=True)
    def data_visualization(self):
        self.data['Make'].str.strip()
        plt.figure(figsize=(10,8))
        sn.boxplot(x=self.data["fuelType"], y=self.data["price"])
        plt.show()
        sn.regplot(x=self.data["year"], y=self.data["price"])
        plt.show()

    def outlier(self):
        for column in self.data.columns[1:-8]:
            for spec in self.data["Make"].unique():
                selected_spec = self.data[self.data["Make"]==spec]
                selected_column = selected_spec[column]
                std = selected_column.std()
                avg = selected_column.mean()
                three_sigma_plus=avg + (3*std)
                three_sigma_minus=(  (3*std)-avg)
                outliers=selected_column[((selected_spec[column] > three_sigma_plus) | (selected_spec[column] < three_sigma_minus))].index
                self.data.drop(index=outliers, inplace=True)
        return self.data
   
    def local_outlier(self):
        self.y = self.data['model']
        self.data.drop(['model'],axis=1,inplace=True)
        self.data.drop(['Make'],axis=1,inplace=True)
        self.clf=LocalOutlierFactor(n_neighbors=20,contamination=0.1)
        self.clf.fit_predict(self.data)
        self.car_new_scores= self.clf.negative_outlier_factor_
        np.sort(self.car_new_scores)[0:3]
        self.esik_deger=np.sort(self.car_new_scores)[2]
        self.aykiri_tf = self.car_new_scores > self.esik_deger
        self.baski_degeri = self.data[self.car_new_scores == self.esik_deger]
        self.aykirilar = self.data[~self.aykiri_tf]
        self.res=self.aykirilar.to_records(index=False)
        self.res[:]=self.baski_degeri.to_records(index=False)
        self.data[~self.aykiri_tf] = pd.DataFrame(self.res,index=self.data[~self.aykiri_tf].index) 
        return self.data[~self.aykiri_tf]

    def min_max_scaler(self):
        self.scaler = MinMaxScaler()
        self.cols=self.data.columns
        self.data = self.scaler.fit_transform(self.data)
        self.data = pd.DataFrame(self.data, columns=[self.cols])
        return self.data

    def train_test(self):        
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.data, self.y, test_size=0.2, random_state=2)
    def logistic_regression(self):
        self.logreg = LogisticRegression(random_state=0,max_iter=120,multi_class='multinomial', solver='lbfgs')
        self.logreg.fit(self.X_train, self.y_train)
        self.y_pred = self.logreg.predict(self.X_test)

    def accuracy_score(self):
        return accuracy_score(self.y_test, self.y_pred)
    def pca(self):
        self.pca=PCA(n_components=7)
        self.data_pca=self.data.copy()
        self.X_pca=self.pca.fit_transform(self.data_pca)
        np.cumsum(np.round(self.pca.explained_variance_ratio_,decimals=4)*100)[0:10]
        self.X_train_pca,self.X_test_pca,self.y_train,self.y_test=train_test_split(self.X_pca,self.y,test_size=0.2,random_state=2)
        self.model=LogisticRegression(max_iter=1000)
        self.model.fit(self.X_train_pca,self.y_train)
    def confusion_matrix(self):
        self.cm=confusion_matrix(self.y_test,self.y_pred)
        plt.figure(figsize=(20,10))
        sn.heatmap(self.cm,annot=True)
        plt.show()
    def pca_score(self):
        return self.model.score(self.X_test_pca,self.y_test)
        

# audi, BMW, Ford, vw, toyota, skoda, Hyundai'dan birini car_name'e yazınız. audi, default değerdir.
def Car(car_name="audi"):
    car=LogisticR(car_name)
    car.feature_engineering()
    car.data_visualization()
    car.dummy()
    car.outlier()
    print(car.local_outlier())
    car.min_max_scaler()
    car.train_test()
    car.logistic_regression()
    car.confusion_matrix()
    print("Modelin Accuracy Score'u: {0:0.4f}".format(car.accuracy_score()))
    car.pca()
    print("Modelin PCA Score'u: {0:0.4f}".format(car.pca_score()))


def main():
    Car("audi")
    Car("BMW")
    Car("Ford")
    Car("vw")
    Car("toyota")
    Car("skoda")
    Car("Hyundai")
main()