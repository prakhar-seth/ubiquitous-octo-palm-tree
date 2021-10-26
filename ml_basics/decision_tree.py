from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn.tree import DecisionTreeClassifier
import pydotplus
iris=datasets.load_iris()
x_train,x_test,y_train,y_test=train_test_split(iris.data,iris.target,random_state=1)
clf1=DecisionTreeClassifier()
clf1.fit(x_train,y_train)
dot_data=export_graphviz(clf1,out_file=None,feature_names=iris.feature_names)
graph=pydotplus.graph_from_dot_data(dot_data)
graph.write("iris.pdf")
y_train_pred=clf1.predict(x_train)
y_test_pred=clf1.predict(x_test)
from sklearn.metrics import confusion_matrix
y_tr=confusion_matrix(y_train,y_train_pred)
y_ts=confusion_matrix(y_test,y_test_pred)
print(y_tr)
print(y_ts)