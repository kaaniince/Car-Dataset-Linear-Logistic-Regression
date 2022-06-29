from re import S
import numpy as np
import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn.preprocessing import MinMaxScaler, scale
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.metrics import r2_score ,mean_squared_error
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix

from msilib.schema import Class
from pydoc import describe
from click import Parameter
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import r2_score
import math
from scipy import stats
from scipy.stats import norm,skew

from sklearn.preprocessing import RobustScaler,StandardScaler,MinMaxScaler
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator,TransformerMixin,RegressorMixin,clone
import math
import warnings
warnings.simplefilter(action='ignore', category=Warning)
data = pd.read_csv('cars_dataset.csv')
cars=data.copy()
class LinearR():
    def __init__(self,data):
        self.data=data
        self.X_train_new = None
    def feature_engineering(self,data):
        self.litre = data.insert(2, 'litre', pd.Series(self.data["mileage"]*(self.data["mpg"]*0.43))/100)
        self.litre = data["litre"]
        return self.litre
    def data_visualization(self):
        cols = ['mileage', 'tax', 'mpg', 'engineSize']
        plt.figure(figsize=(20,25))
        for i in range(len(cols)):
            plt.subplot(5,3,i+1)
            plt.title(cols[i] + ' - Price')
            sn.regplot(x=eval('data' + '.' + cols[i]), y=self.data["price"])
        plt.tight_layout()
        self.data['Make'].str.strip()
        plt.figure(figsize=(10,8))
        sn.boxplot(x=self.data["fuelType"], y=self.data["price"])
        plt.show()
    def dummy(self):
        dummies_list=['transmission','fuelType','model']
        for i in dummies_list:
            temp_df = pd.get_dummies(eval('cars' + '.' + i), drop_first=True)
            self.data = pd.concat([self.data, temp_df], axis=1)
            self.data.drop([i], axis=1, inplace=True)
        return self.data 
    def feature_delete(self):
        self.data.drop(['Make'],axis=1,inplace=True)
    def train_test(self):
        np.random.seed(0)
        self.df_train, self.df_test = train_test_split(self.data, train_size = 0.7, test_size = 0.3, random_state = 100)   
    def min_max_scaler(self):
        self.scaler = MinMaxScaler()
        scale_cols = ['year','engineSize','litre','tax','price']
        self.df_train[scale_cols] = self.scaler.fit_transform(self.df_train[scale_cols])
        self.y_train = self.df_train.pop('price')
        self.X_train = self.df_train
    def linear_regression(self):
        self.lm = LinearRegression()
        self.lm.fit(self.X_train,self.y_train)
        rfe = RFE(self.lm, n_features_to_select=10)
        rfe = rfe.fit(self.X_train, self.y_train)
        self.X_train_rfe = self.X_train[self.X_train.columns[rfe.support_]]
    def ols_build(self,X,y):
        X = sm.add_constant(X) 
        self.lm = sm.OLS(y,X).fit() 
        print(self.lm.summary()) 
        return X
    def ols_error(self):
        self.lm = sm.OLS(self.y_train,self.X_train_new).fit()
        y_train_price = self.lm.predict(self.X_train_new)
        fig = plt.figure()
        sn.distplot((self.y_train - y_train_price), bins = 20)
        fig.suptitle('Error Terms', fontsize = 20) 
        plt.xlabel('Errors', fontsize = 18) 
        plt.show()
    def regression_pred_test(self):
        num_vars = ['year','engineSize','litre','tax','price']
        self.df_test[num_vars] = self.scaler.fit_transform(self.df_test[num_vars])
        self.y_test = self.df_test.pop('price')
        self.y_test = self.y_test.values.reshape(-1, 1)
        self.X_test = self.df_test
        self.X_train_new = self.X_train_new.drop('const',axis=1)
        self.X_test_new = self.X_test[self.X_train_new.columns]
        self.X_test_new = sm.add_constant(self.X_test_new)
        self.y_pred = self.lm.predict(self.X_test_new)
        self.y_pred = self.y_pred.values.reshape(-1, 1)
        return self.y_test,self.y_pred
    
    
    def r2_score(self):
        return r2_score(self.y_test, self.y_pred)
    def coefficients(self):
        self.X_train_final =self. X_train_new[['year', 'engineSize', 'litre',' 8 Series',' California', ' R8',' X7',' i3',' i8']]
        self.X_train_final.columns
        self.lr_final = LinearRegression()
        self.lr_final.fit(self.X_train_final, self.y_train)
        self.coefficient = pd.DataFrame(self.lr_final.coef_, index = ['year', 'engineSize', 'litre',' 8 Series',' California', ' R8',' X7',' i3',' i8'], 
        columns=['Coefficient'])
        return self.coefficient.sort_values(by=['Coefficient'], ascending=False)
    def rmse(self):
        self.mse = mean_squared_error(self.y_test, self.y_pred)
        self.rmse = math.sqrt(self.mse)
        return self.rmse
    def pca(self):
        scale_cols = ['year','engineSize','litre','tax','mileage','mpg',' 8 Series',' California', ' R8',' X7',' i3',' i8']
        self.data_pca=self.data.copy()
        self.y = self.data_pca.pop('price')
        self.data_pca[scale_cols] = self.scaler.fit_transform(self.data_pca[scale_cols])
        self.pca=PCA(n_components=53)
        self.X_pca=self.pca.fit_transform(self.data_pca)
        np.cumsum(np.round(self.pca.explained_variance_ratio_,decimals=4)*100)[0:10]
        self.X_train_pca,self.X_test_pca,self.y_train,self.y_test=train_test_split(self.X_pca,self.y,test_size=0.2,random_state=2)
        model=LinearRegression()
        model.fit(self.X_train_pca,self.y_train)
        self.y_pred_pca = model.predict(self.X_test_pca)
        return r2_score(self.y_test,self.y_pred_pca)
    def plot(self):
        plt.subplots(figsize=(12,8))
        plt.scatter(range(len(self.y_test)), self.y_test, color='blue')
        plt.scatter(range(len(self.y_pred)), self.y_pred, color='red')
        plt.show()
def main():
    prediction_price = LinearR(cars)
    prediction_price.feature_engineering(cars)
    prediction_price.data_visualization()
    prediction_price.dummy()
    prediction_price.feature_delete()
    prediction_price.train_test()
    prediction_price.min_max_scaler()
    prediction_price.linear_regression()
    prediction_price.X_train_new=prediction_price.ols_build(prediction_price.X_train_rfe,prediction_price.y_train)
    prediction_price.ols_error()
    prediction_price.regression_pred_test()
    print(f"R'2:{prediction_price.r2_score()}")
    print(f"Katsayılar:{prediction_price.coefficients()}")
    print(f"RMSE:{prediction_price.rmse()}")
    print(f"PCA R'2:{prediction_price.pca()}")
    prediction_price.plot()
    
main()
#car2 = Linear(car.data)
#print(car2.data)
