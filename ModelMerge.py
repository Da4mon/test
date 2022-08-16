from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
# from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso
# from catboost import CatBoostRegressor
from sklearn.ensemble import  RandomForestRegressor
# from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR, SVC
from sklearn.model_selection import KFold, train_test_split
from sklearn.linear_model import LogisticRegression
import pandas as pd
import numpy as np
import warnings


# 定义梯度提升模型+第二层模型
class GradientBoostPlusModel:
    def __init__(self, trees_model, second_model):
        self.trees_model = trees_model
        self.second_model = second_model
        self.leaf_features = None

    def fit(self, X, y):
        self.trees_model.fit(X=X, y=y)  # 拟合第一层的树模型
        try:  # GBDT/XGB
            leaf_X = self.trees_model.apply(X)
        except AttributeError:  # LGB
            leaf_X = self.trees_model.predict(X, pred_leaf=True)

        self.second_model.fit(X=leaf_X, y=y)  # 拟合第二层的模型

        pass

    def predict(self, X):
        try:
            leaf_data_X = self.trees_model.apply(X)
        except AttributeError:
            leaf_data_X = self.trees_model.predict(X, pred_leaf=True)

        pre_result = self.second_model.predict(X=leaf_data_X)

        # 预测结果
        return pre_result


# 定义模型Stacking
class StackModels:
    def __init__(self, first_layer_model, second_model, n_splits):
        self.first_layer_model, self.second_model = first_layer_model, second_model
        self.n_splits = n_splits
        self.fold, self.fold_model = None, [[] for _ in range(len(first_layer_model))]
        self.second_layer_features = None

    def fit(self, X, y):
        self.fold = KFold(n_splits=self.n_splits, shuffle=True, random_state=100)  # 交叉验证器
        self.second_layer_features = pd.DataFrame(data=np.random.uniform(size=[X.shape[0],
                                                                               len(self.first_layer_model)]),
                                                  columns=[f"model_train_{i + 1}"
                                                           for i in range(len(self.first_layer_model))])

        # 拟合第一层的各个模型，并生成特征转换层
        for temp_model_index, temp_model in enumerate(self.first_layer_model):
            # 对于数据的每一折循环
            for train_index, val_index in self.fold.split(X):
                temp_model.fit(X.iloc[train_index, :], y[train_index])  # 拟合模型
                self.fold_model[temp_model_index].append(temp_model)  # 将训练好的模型保存至列表

                # 将拟合好的模型预测余下一折，并将其作为下一层模型的特征
                self.second_layer_features.iloc[val_index,
                                                temp_model_index] = temp_model.predict(X.iloc[val_index, :])

        # 拟合第二层的模型
        self.second_model.fit(self.second_layer_features, y)

        pass

    def predict(self, X):
        second_layer_features = pd.DataFrame()
        # 用第一层的各个模型预测并取平均值，作为测试集的第二层特征
        for temp_single_model_list in self.fold_model:

            # 循环输出单种模型不同折的拟合结果
            single_model_pre_mean = np.zeros(shape=[X.shape[0], ])
            for temp_single_model in temp_single_model_list:
                temp_pre = temp_single_model.predict(X)
                single_model_pre_mean += temp_pre

            single_model_pre_mean /= len(self.first_layer_model)
            second_layer_features = pd.concat([second_layer_features,
                                               pd.DataFrame(single_model_pre_mean)], axis=1)

        second_layer_features.columns = [f"model_train_{i + 1}" for i in range(len(self.first_layer_model))]
        final_pre_result = self.second_model.predict(second_layer_features)

        return final_pre_result


# 定义模型Blending
class BlendModels:
    def __init__(self, first_layer_model, second_model, blend_size):
        self.first_layer_model, self.second_model = first_layer_model, second_model
        self.X_train, self.X_val, self.y_train, self.y_val = None, None, None, None,
        self.fold, self.fold_model = None, []
        self.second_layer_features = None  # 第二层的训练特征
        self.blend_size = blend_size  # 测试集的百分比

    def fit(self, X, y):
        self.X_train, self.X_val, self.y_train, self.y_val = \
            train_test_split(X, y, train_size=(1 - self.blend_size),
                             test_size=self.blend_size, shuffle=True, random_state=100)  # 留出法验证器

        # 初始化第二层的模型特征
        self.second_layer_features = pd.DataFrame(data=np.random.uniform(size=[self.X_val.shape[0],
                                                                               len(self.first_layer_model)]),
                                                  columns=[f"model_train_{i + 1}"
                                                           for i in range(len(self.first_layer_model))])
        # 拟合第一层的各个模型，并生成特征转换层
        for temp_model_index, temp_model in enumerate(self.first_layer_model):
            temp_model.fit(self.X_train, self.y_train)  # 拟合模型
            self.fold_model.append(temp_model)  # 将训练好的模型保存至列表

            # 将拟合好的模型预测留出数据集，并将其作为下一层模型的特征
            self.second_layer_features.iloc[:, temp_model_index] = temp_model.predict(self.X_val)

            # 拟合第二层的模型
        self.second_model.fit(self.second_layer_features, self.y_val)

        pass

    def predict(self, X):
        second_layer_features = pd.DataFrame()
        # 用第一层的各个模型预测并取平均值，作为测试集的第二层特征
        for temp_single_model in self.fold_model:
            # 循环输出单种模型的拟合结果
            temp_pre = temp_single_model.predict(X)
            second_layer_features = pd.concat([second_layer_features,
                                               pd.DataFrame(temp_pre)], axis=1)

        second_layer_features.columns = [f"model_train_{i + 1}" for i in range(len(self.first_layer_model))]
        final_pre_result = self.second_model.predict(second_layer_features)

        return final_pre_result


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    # 一、读取数据
    raw_train_data = pd.read_csv("../raw_data/zhengqi_train.txt", sep="\t")
    raw_test_data = pd.read_csv("../raw_data/zhengqi_test.txt", sep="\t")

    # 二、BoostTree+第二层模型
    gbpm = GradientBoostPlusModel(trees_model=XGBRegressor(),
                                  second_model=Ridge(normalize=True))
    gbpm.fit(X=raw_train_data.iloc[:, :-1], y=raw_train_data.iloc[:, -1])
    pre_train_y = gbpm.predict(X=raw_train_data.iloc[:, :-1])
    mse = mean_squared_error(y_true=raw_train_data.iloc[:, -1], y_pred=pre_train_y)
    r2 = r2_score(y_true=raw_train_data.iloc[:, -1], y_pred=pre_train_y)
    print(f"MSE: {mse} R2: {r2}\n")

    # 三、模型Stacking
    my_first_layer_model = [RandomForestRegressor(), XGBRegressor(), SVR(), LGBMRegressor(), KNeighborsRegressor()]
    sm = StackModels(first_layer_model=my_first_layer_model,
                     second_model=Ridge(), n_splits=3)
    sm.fit(X=raw_train_data.iloc[:, :-1], y=raw_train_data.iloc[:, -1])
    pre_train_y = sm.predict(X=raw_train_data.iloc[:, :-1])
    mse = mean_squared_error(y_true=raw_train_data.iloc[:, -1], y_pred=pre_train_y)
    r2 = r2_score(y_true=raw_train_data.iloc[:, -1], y_pred=pre_train_y)
    print(f"MSE: {mse} R2: {r2}\n")

    # 四、模型Blending
    # my_first_layer_model = [RandomForestRegressor(), XGBRegressor(), SVR(), LGBMRegressor(), KNeighborsRegressor()]
    my_first_layer_model = [RandomForestRegressor(), XGBRegressor(), KNeighborsRegressor()]
    bm = BlendModels(first_layer_model=my_first_layer_model,
                     second_model=Ridge(), blend_size=0.3)
    bm.fit(X=raw_train_data.iloc[:, :-1], y=raw_train_data.iloc[:, -1])
    pre_train_y = bm.predict(X=raw_train_data.iloc[:, :-1])
    mse = mean_squared_error(y_true=raw_train_data.iloc[:, -1], y_pred=pre_train_y)
    r2 = r2_score(y_true=raw_train_data.iloc[:, -1], y_pred=pre_train_y)
    print(f"MSE: {mse} R2: {r2}\n")
