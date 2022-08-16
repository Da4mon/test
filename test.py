import pandas as pd
from sklearn import datasets  # 导入库
from ModelMerge import BlendModels
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

cancer = datasets.load_breast_cancer(as_frame=True)  # 导入乳腺癌数据
raw_data = cancer["frame"]
X, y = raw_data.drop(labels=["target"], axis=1), raw_data["target"]

BM = BlendModels(first_layer_model=[SVC()], second_model=LogisticRegression(), blend_size=0.3)
BM.fit(X, y)
BM.predict(X)
