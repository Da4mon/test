from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from itertools import combinations_with_replacement
import pandas as pd


# 强相关系数特征两两交叉组合
class CorrCrossFeatures:
    def __init__(self, corr_threshold, corr_method="pearson"):
        """
        :param corr_threshold: 根据相关系数的阈值选择特征，大于等于此阈值绝对值的将被选择进行交叉
        :param corr_method: 选择特征的相关性计算方式
        """
        self.corr_threshold, self.corr_dict, self.corr_method = corr_threshold, None, corr_method
        self.best_feature_names, self.transformed_X = None, None

    def fit(self, X, y):
        # 生成各变量X与y的相关系数字典
        self.corr_dict = dict(
            X[X.columns].corrwith(y, axis="index", method=self.corr_method)
        )
        # 选择相关系数大于阈值的特征
        self.best_feature_names = [col_name for col_name in self.corr_dict.keys()
                                   if abs(self.corr_dict[col_name]) >= self.corr_threshold]
        pass

    def transform(self, X):
        # 抽取组合特征名
        for features_combination in combinations_with_replacement(self.best_feature_names, 2):
            features_first, features_second = features_combination[0], features_combination[1]
            # 计算交叉特征
            X[f"{features_first}*{features_second}"] = X[features_first] * X[features_second]

        return X

    def fit_transform(self, X, y):
        self.fit(X=X, y=y)
        self.transformed_X = self.transform(X=X)

        return self.transformed_X


# K均值聚类特征生成
class KmeansClusterFeatures:
    def __init__(self, kmeans_model):
        """
        :param kmeans_model: k均值聚类的模型
        """
        self.kmeans_model = kmeans_model
        self.transformed_X = None

    def fit(self, X):
        self.kmeans_model.fit(X=X)

        pass

    def transform(self, X):
        cluster_classes = self.kmeans_model.predict(X=X)
        X["kmeans_classes"] = cluster_classes

        return X

    def fit_transform(self, X):
        self.fit(X=X)
        self.transformed_X = self.transform(X=X)

        return self.transformed_X


if __name__ == "__main__":
    # 一、读取数据
    raw_train_data = pd.read_csv("../raw_data/zhengqi_train.txt", sep="\t")
    raw_test_data = pd.read_csv("../raw_data/zhengqi_test.txt", sep="\t")

    ccs = CorrCrossFeatures(corr_threshold=0.7, corr_method="pearson")
    ccs_transformed_train_X = ccs.fit_transform(X=raw_train_data.iloc[:, :-1], y=raw_train_data["target"])
    ccs.transform(raw_test_data)

    kcf = KmeansClusterFeatures(kmeans_model=KMeans(n_clusters=8))
    kcf.fit_transform(X=raw_train_data.iloc[:, :-1])
    kcf.transform(X=raw_test_data)
