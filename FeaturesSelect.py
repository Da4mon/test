from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import pandas as pd
# import numpy as np


# 随机森林特征选择
class RandomForestSelection:
    def __init__(self, train_df, target_name, n_best, target_type, args_dict=None):
        """
        :param train_df: 输入的训练集DataFrame
        :param target_name: 输入的训练集标签名
        :param n_best: 选择最重要的特征个数
        :param args_dict: 随机森林模型的参数字典
        """
        self.train_X, self.train_y = train_df.drop(labels=[target_name], axis=1), train_df[target_name]
        self.feature_names, self.target_name = list(self.train_X.columns), self.train_y.name
        self.n_best = n_best
        if target_type == "regression":
            self.model = RandomForestRegressor(**args_dict) if args_dict else RandomForestRegressor()
        elif target_type == "classification":
            self.model = RandomForestClassifier(**args_dict) if args_dict else RandomForestClassifier()
        else:
            print("任务类型选择错误")

        self.best_features_index, self.transformed_train_X = None, None

    def fit_transform(self):
        self.model.fit(X=self.train_X, y=self.train_y)  # 拟合模型

        # 选择最重要的n_best个特征排名
        self.best_features_index = self.model.feature_importances_.argsort()[-1:-(self.n_best + 1):-1]
        self.transformed_train_X = self.train_X.iloc[:, self.best_features_index]

        return self.transformed_train_X


# 相关系数特征选择
class CorrSelection:
    def __init__(self, train_df, target_name, corr_threshold, corr_method="pearson"):
        """
        :param train_df: 输入的训练集DataFrame
        :param target_name: 输入的训练集标签名
        :param corr_threshold: 根据相关系数的阈值选择特征，大于等于此阈值绝对值的将被选择
        :param corr_method: 相关系数的计算方式
        """
        self.train_X, self.train_y = train_df.drop(labels=[target_name], axis=1), train_df[target_name]
        self.feature_names, self.target_name = list(self.train_X.columns), self.train_y.name
        self.corr_threshold = corr_threshold
        self.corr_dict, self.corr_method = None, corr_method
        self.best_feature_names, self.transformed_train_X = None, None

    def fit_transform(self):
        self.corr_dict = dict(
            self.train_X[self.feature_names].corrwith(self.train_y,
                                                      axis="index", method=self.corr_method)
        )
        # 选择相关系数最高的特征
        self.best_feature_names = [col_name for col_name in self.corr_dict.keys()
                                   if abs(self.corr_dict[col_name]) >= self.corr_threshold]
        self.transformed_train_X = self.train_X[self.best_feature_names]

        return self.transformed_train_X


if __name__ == "__main__":
    # 一、读取数据
    raw_train_data = pd.read_csv("../raw_data/zhengqi_train.txt", sep="\t")
    raw_test_data = pd.read_csv("../raw_data/zhengqi_test.txt", sep="\t")

    rfs = RandomForestSelection(train_df=raw_train_data, target_name="target",n_best=10, target_type="regression",
                                args_dict={"n_estimators": 200, "n_jobs": 5})
    rfs_transformed_train_X = rfs.fit_transform()
    print(len(rfs.model.estimators_))

    crs = CorrSelection(train_df=raw_train_data, target_name="target", corr_threshold=0.4, corr_method="pearson")
    # crs = CorrSelection(train_df=raw_train_data, target_name="target", corr_threshold=0.4, corr_method="kendall")
    crs_transformed_train_X = crs.fit_transform()








