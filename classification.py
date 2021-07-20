import pandas
from sklearn import tree
import numpy
import missingno
import pydotplus
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import seaborn as sn
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from imblearn.metrics import sensitivity_score
from imblearn.metrics import specificity_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import os
import pydot
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image
import graphviz
def get_dataset():
    df=pandas.read_csv("Iris.csv")
    print("\n-----Attributes & it's types------\n")
    print(df.dtypes)
    print("\nTotal number of records = ",len(df))
    return df
def check_missing_values(df):
    missingno.bar(df)
    #plt.show()

def data_cleaning(df):
    #data cleaning is an important step for data analysis.
    #Now we know that our data has no missing values as each column of the bar has same length.
    #We have five features - Id,Sepal Length, Sepal Width, Petal Length, and Petal Width and one target field :
    # Species.One column (Id) is of no use for us. So now we will drop that column permanently(using inplace= True).
    df.drop(['Id'],axis=1,inplace=True)
    df.columns=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm','Species']
    #print(df)
    #We do not have any nulls, but let's quickly look at the features again to double check to see if there are any obvious outliers,
    # we can box plot our features.
    df.plot(kind='box',subplots=True,layout=(2,2))
    plt.suptitle("Box Plot")
    #plt.show()
    #we realized SepalWidthCm has outliers
    #To remove the outliers there are two solutions either we can remove the corresponding record or replace it by some suitable values
    # removing the record may incur loss of data which may be valuable
    #thus we will replace it by suitable values
    #by suitable values we mean the outliers which are greater tha Q3 we will replace them by Q3
    #and the outliers which are smaller than Q1 we will replace them by Q1
    print("Box Plot description of SepalWidthCm")
    print(df['SepalWidthCm'].describe())
    df['SepalWidthCm']=numpy.where(df['SepalWidthCm']>=3.30,3.30,df['SepalWidthCm'])
    df['SepalWidthCm']=numpy.where(df['SepalWidthCm']<=2.80,2.80,df['SepalWidthCm'])
    #print(df)

    df.plot(kind='box',subplots=True,layout=(2,2))
    plt.suptitle("After removal of outliers")
    #plt.show()
def split_data(df):
    x = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
    print("\nThe Data set is divided into 80:20")
    print(X_train.shape,y_train.shape)
    print(X_test.shape,y_test.shape)
    return X_train,X_test,y_train,y_test

def train_using_gini(X_train, X_test, y_train):
    clf_gini = DecisionTreeClassifier(criterion = "gini")
    clf_gini.fit(X_train, y_train)
    return clf_gini

def naive_bayes(X_test,X_train,y_train):
    gnb=GaussianNB()
    gnb.fit(X_train, y_train)
    # making predictions on the testing set
    y_pred = gnb.predict(X_test)
    return y_pred

def train_using_entropy(X_train,X_test,y_train):
    clf_entropy = DecisionTreeClassifier(criterion = "entropy")

    # Performing training
    clf_entropy.fit(X_train, y_train)
    return clf_entropy

def prediction(X_test,clf_object):
    y_prediction = clf_object.predict(X_test)
    return y_prediction

def correctness(y_test,y_pred):
    matrix=confusion_matrix(y_test,y_pred)
    print("Accuracy rate =",metrics.accuracy_score(y_test, y_pred))
    print("Error rate = ",1-metrics.accuracy_score(y_test, y_pred))
    print("Sensitivity score = ",sensitivity_score(y_test, y_pred, average='weighted'))
    print("Specifity Score = ",specificity_score(y_test, y_pred, average='weighted'))
    print("Classification Report = \n",classification_report(y_test,y_pred))
    print("Confusion Matrix = ",matrix)
    df_cm=pandas.DataFrame(matrix,index = [i for i in ['setosa','versicolor','virginica']],
                  columns = [i for i in ['setosa','versicolor','virginica']])
    plt.figure(figsize = (12,10))
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    sn.heatmap(df_cm, annot=True)
    plt.show()

def imbalance_problem(df):
    df['Species'].value_counts().plot.pie(explode=[0.1,0.1,0.1],autopct='%1.1f%%',shadow=True,figsize=(10,8))
    plt.show()

def decision_tree(df,X_train,y_train):
    '''
    os.environ["PATH"] +=os.pathsep + r"C:\logs\Python 3.8.1\Lib\site-packages\graphviz"
    features=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
    d_tree=DecisionTreeClassifier()
    clf=d_tree.fit(X_train,y_train)
    dot_data=StringIO()
    raw=tree.export_graphviz(clf,out_file=None,feature_names=features,filled=True,rounded=True,special_characters=True,class_names=['Iris-setosa','Iris-versicolor','Iris-virginica'])
    graph=pydotplus.graph_from_dot_data(raw)
    graph.write_png('Decision_Tree.png')
    img=pltimg.imread('Decision_Tree.png')
    imgplot=plt.imshow(img)
    plt.show()'''
    '''
    os.environ["PATH"] +=os.pathsep + r"C:\logs\Python 3.8.1\Lib\site-packages\graphviz"
    features=['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']
    d_tree=DecisionTreeClassifier()
    clf=d_tree.fit(X_train,y_train)
    dot_data=StringIO()
    export_graphviz(clf,out_file=dot_data,feature_names=features,filled=True,rounded=True,special_characters=True,class_names=['Iris-setosa','Iris-versicolor','Iris-virginica'])
    graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_png('decision_tree.png')
    Image(graph.create_png())
    '''
    os.environ["PATH"] +=os.pathsep + r"C:\logs\Python 3.8.1\Lib\site-packages\graphviz"
    d_tree=DecisionTreeClassifier()
    d_tree=d_tree.fit(X_train,y_train)
    features=list(df.columns[0:-1])
    #print(features)
    dot_data=StringIO()
    export_graphviz(d_tree,out_file=dot_data,feature_names=features,filled=True,rounded=True)
    graph=pydotplus.graph_from_dot_data(dot_data.getvalue()).write_png('decision_tree.png')
    #Image(graph.create_png())
    img=pltimg.imread('Decision_Tree.png')
    imgplot=plt.imshow(img)
    plt.show()

def main():
    df=get_dataset()
    check_missing_values(df)
    data_cleaning(df)
    X_train,X_test,y_train,y_test=split_data(df)
    '''
    print("\n-----According to Naive Bayes Classification----\n")
    y_pred=naive_bayes(X_test,X_train,y_train)
    correctness(y_test,y_pred)

    print("\n----------Using Entropy------------------------\n")
    clf_entropy=train_using_entropy(X_train,X_test,y_train)
    y_predict_entropy=prediction(X_test,clf_entropy)
    correctness(y_test,y_predict_entropy)

    #prediction using gini method
    print("\n---------Using Gini Index Method----------------\n")
    clf_gini=train_using_gini(X_train,X_test,y_train)
    y_predict_gini = prediction(X_test, clf_gini)
    correctness(y_test, y_predict_gini)

    imbalance_problem(df)
    '''
    decision_tree(df,X_train,y_train)
if __name__=="__main__":
    main()
