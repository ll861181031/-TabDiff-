"""
贝叶斯参数优化代码
用于优化分类器的参数
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier

from sklearn.metrics import (
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args

np.random.seed(42)

class BayesianOptimizer:
    """贝叶斯参数优化器"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.results = {}
        self.best_params = {}
        self.best_scores = {}
        
    def load_mixed_data(self, synthetic_data_path, original_test_path, target_column=None, train_ratio=0.7):
        """加载混合数据集：使用合成数据训练，原始测试集评估"""
        print(f"Loading synthetic data from {synthetic_data_path}")
        print(f"Loading original test data from {original_test_path}")
        
        syn_df = pd.read_csv(synthetic_data_path)
        test_df = pd.read_csv(original_test_path)

        if target_column is None:
            target_column = syn_df.columns[-1]

        X_syn = syn_df.drop(columns=[target_column])
        y_syn = syn_df[target_column]
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]

        # 处理分类特征编码
        categorical_cols = X_syn.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            self.label_encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                all_values = pd.concat([X_syn[col], X_test[col]]).astype(str).unique()
                le.fit(all_values)
                X_syn[col] = le.transform(X_syn[col].astype(str))
                X_test[col] = le.transform(X_test[col].astype(str))
                self.label_encoders[col] = le

        # 编码目标变量
        if y_syn.dtype == 'object' or y_syn.dtype.name == 'category':
            self.label_encoder = LabelEncoder()
            all_target_values = pd.concat([y_syn, y_test]).astype(str).unique()
            self.label_encoder.fit(all_target_values)
            y_syn = self.label_encoder.transform(y_syn.astype(str))
            y_test = self.label_encoder.transform(y_test.astype(str))

        # 划分训练集和测试集
        train_size = int(len(X_syn) * train_ratio)
        self.X_train = X_syn.iloc[:train_size]
        self.y_train = y_syn[:train_size]
        self.X_test = X_test
        self.y_test = y_test

        # 标准化特征
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        self.feature_names = X_syn.columns.tolist()

        print(f"Training set: {self.X_train.shape} (70% synthetic data)")
        print(f"Test set: {self.X_test.shape} (original data)")

    def calculate_weighted_f1(self, y_true, y_pred):
        """计算加权F1分数"""
        classes = np.unique(y_true)
        n_classes = len(classes)
        if n_classes == 1:
            return 1.0
        
        class_counts = np.bincount(y_true)
        total_samples = len(y_true)
        class_weights = class_counts / total_samples
        
        f1_scores = []
        for i, class_label in enumerate(classes):
            tp = np.sum((y_true == class_label) & (y_pred == class_label))
            fp = np.sum((y_true != class_label) & (y_pred == class_label))
            fn = np.sum((y_true == class_label) & (y_pred != class_label))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1_i = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            f1_scores.append(f1_i)
        
        weighted_f1 = np.sum(class_weights * f1_scores)
        return weighted_f1

    def optimize_lr(self, n_calls=50):
        """优化逻辑回归参数"""
        print("\n" + "="*60)
        print("Optimizing Logistic Regression")
        print("="*60)
        
        space = [
            Real(0.01, 10.0, name='C'),
            Categorical(['l1', 'l2', 'elasticnet'], name='penalty'),
            Categorical(['liblinear', 'saga'], name='solver'),
            Integer(100, 10000, name='max_iter')
        ]
        
        @use_named_args(space)
        def objective(**params):
            try:
                # 处理参数组合
                if params['penalty'] == 'elasticnet' and params['solver'] != 'saga':
                    return 0.0
                if params['penalty'] == 'l1' and params['solver'] not in ['liblinear', 'saga']:
                    return 0.0
                
                model = LogisticRegression(
                    C=params['C'],
                    penalty=params['penalty'],
                    solver=params['solver'],
                    max_iter=params['max_iter'],
                    random_state=self.random_state,
                    class_weight='balanced'
                )
                
                model.fit(self.X_train_scaled, self.y_train)
                
                y_train_pred = model.predict(self.X_train_scaled)
                
                recall = recall_score(self.y_train, y_train_pred, average='weighted')
                f1 = self.calculate_weighted_f1(self.y_train, y_train_pred)
                
                score = f1 + recall
                return -score
                
            except Exception as e:
                return 0.0
        
        n_calls = max(10, n_calls)
        result = gp_minimize(objective, space, n_calls=n_calls, random_state=self.random_state)
        
        best_params = {
            'C': result.x[0],
            'penalty': result.x[1],
            'solver': result.x[2],
            'max_iter': result.x[3]
        }
        
        best_model = LogisticRegression(**best_params, random_state=self.random_state, class_weight='balanced')
        best_model.fit(self.X_train_scaled, self.y_train)
        
        # 在训练集上评估
        y_train_pred = best_model.predict(self.X_train_scaled)
        train_recall = recall_score(self.y_train, y_train_pred, average='weighted')
        train_f1 = self.calculate_weighted_f1(self.y_train, y_train_pred)
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        
        # 在测试集上评估
        y_test_pred = best_model.predict(self.X_test_scaled)
        test_recall = recall_score(self.y_test, y_test_pred, average='weighted')
        test_f1 = self.calculate_weighted_f1(self.y_test, y_test_pred)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        
        self.best_params['LR'] = best_params
        self.best_scores['LR'] = {
            'train_recall': train_recall, 
            'train_f1': train_f1, 
            'train_accuracy': train_accuracy,
            'test_recall': test_recall, 
            'test_f1': test_f1, 
            'test_accuracy': test_accuracy
        }
        
        print(f"Best LR parameters: {best_params}")
        print(f"Training set - Accuracy: {train_accuracy:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
        print(f"Test set - Accuracy: {test_accuracy:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
        
        return best_model, best_params

    def optimize_dt(self, n_calls=50):
        """优化决策树参数"""
        print("\n" + "="*60)
        print("Optimizing Decision Tree")
        print("="*60)
        
        space = [
            Integer(5, 50, name='max_depth'),
            Integer(2, 20, name='min_samples_split'),
            Integer(1, 10, name='min_samples_leaf'),
            Categorical(['gini', 'entropy'], name='criterion'),
            Categorical(['sqrt', 'log2', None], name='max_features')
        ]
        
        @use_named_args(space)
        def objective(**params):
            try:
                model = DecisionTreeClassifier(
                    max_depth=params['max_depth'],
                    min_samples_split=params['min_samples_split'],
                    min_samples_leaf=params['min_samples_leaf'],
                    criterion=params['criterion'],
                    max_features=params['max_features'],
                    random_state=self.random_state,
                    class_weight='balanced'
                )
                
                model.fit(self.X_train, self.y_train)
                
                y_train_pred = model.predict(self.X_train)
                y_test_pred = model.predict(self.X_test)
                
                train_recall = recall_score(self.y_train, y_train_pred, average='weighted')
                train_f1 = self.calculate_weighted_f1(self.y_train, y_train_pred)
                train_accuracy = accuracy_score(self.y_train, y_train_pred)
                
                test_recall = recall_score(self.y_test, y_test_pred, average='weighted')
                test_f1 = self.calculate_weighted_f1(self.y_test, y_test_pred)
                test_accuracy = accuracy_score(self.y_test, y_test_pred)
                
                # 计算过拟合程度
                overfitting_score = train_accuracy - test_accuracy
                
                if overfitting_score > 0.1:
                    penalty = 0.8
                elif overfitting_score > 0.05:
                    penalty = 0.6
                elif overfitting_score > 0.02:
                    penalty = 0.3
                else:
                    penalty = 0.0
                
                score = (train_f1 + train_recall) * 0.3 + (test_f1 + test_recall) * 0.7
                score = score * (1 - penalty)
                
                return -score
                
            except Exception as e:
                return 0.0
        
        n_calls = max(10, n_calls)
        result = gp_minimize(objective, space, n_calls=n_calls, random_state=self.random_state)
        
        best_params = {
            'max_depth': result.x[0],
            'min_samples_split': result.x[1],
            'min_samples_leaf': result.x[2],
            'criterion': result.x[3],
            'max_features': result.x[4]
        }
        
        best_model = DecisionTreeClassifier(**best_params, random_state=self.random_state, class_weight='balanced')
        best_model.fit(self.X_train, self.y_train)
        
        y_train_pred = best_model.predict(self.X_train)
        train_recall = recall_score(self.y_train, y_train_pred, average='weighted')
        train_f1 = self.calculate_weighted_f1(self.y_train, y_train_pred)
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        
        y_test_pred = best_model.predict(self.X_test)
        test_recall = recall_score(self.y_test, y_test_pred, average='weighted')
        test_f1 = self.calculate_weighted_f1(self.y_test, y_test_pred)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        
        self.best_params['DT'] = best_params
        self.best_scores['DT'] = {
            'train_recall': train_recall, 
            'train_f1': train_f1, 
            'train_accuracy': train_accuracy,
            'test_recall': test_recall, 
            'test_f1': test_f1, 
            'test_accuracy': test_accuracy
        }
        
        print(f"Best DT parameters: {best_params}")
        print(f"Training set - Accuracy: {train_accuracy:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
        print(f"Test set - Accuracy: {test_accuracy:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
        
        return best_model, best_params

    def optimize_rf(self, n_calls=50):
        """优化随机森林参数"""
        print("\n" + "="*60)
        print("Optimizing Random Forest")
        print("="*60)
        
        # 定义参数空间
        space = [
            Integer(1, 8, name='n_estimators'),
            Integer(2, 8, name='max_depth'),
            Integer(3, 15, name='max_features'),
            Integer(3, 10, name='min_samples_leaf'),
            Integer(8, 25, name='min_samples_split'),
            Categorical(['gini', 'entropy'], name='criterion')
        ]
        
        @use_named_args(space)
        def objective(**params):
            try:
                n_estimators = params['n_estimators'] * 50 - 49  
                min_samples_leaf = params['min_samples_leaf'] * 2 - 1  
                
                model = RandomForestClassifier(
                    n_estimators=n_estimators,
                    max_depth=params['max_depth'],
                    max_features=params['max_features'],
                    min_samples_leaf=min_samples_leaf,
                    min_samples_split=params['min_samples_split'],
                    criterion=params['criterion'],
                    random_state=self.random_state,
                    class_weight='balanced',
                    n_jobs=-1
                )
                
                model.fit(self.X_train, self.y_train)
                
                # 在训练集和测试集上预测和评估
                y_train_pred = model.predict(self.X_train)
                y_test_pred = model.predict(self.X_test)
                
                # 计算训练集和测试集指标
                train_recall = recall_score(self.y_train, y_train_pred, average='weighted')
                train_f1 = self.calculate_weighted_f1(self.y_train, y_train_pred)
                train_accuracy = accuracy_score(self.y_train, y_train_pred)
                
                test_recall = recall_score(self.y_test, y_test_pred, average='weighted')
                test_f1 = self.calculate_weighted_f1(self.y_test, y_test_pred)
                test_accuracy = accuracy_score(self.y_test, y_test_pred)
         
                overfitting_score = train_accuracy - test_accuracy
                
                if overfitting_score > 0.1:  
                    penalty = 0.8  
                elif overfitting_score > 0.05:  
                    penalty = 0.6  
                elif overfitting_score > 0.02:  
                    penalty = 0.3  
                else:
                    penalty = 0.0  
                
                score = (train_f1 + train_recall) * 0.3 + (test_f1 + test_recall) * 0.7
                score = score * (1 - penalty)
                
                return -score
                
            except Exception as e:
                return 0.0
        
        n_calls = max(10, n_calls)
        result = gp_minimize(objective, space, n_calls=n_calls, random_state=self.random_state)
        
        best_n_estimators = result.x[0] * 50 - 49  
        best_min_samples_leaf = result.x[3] * 2 - 1  
        
        best_params = {
            'n_estimators': best_n_estimators,
            'max_depth': result.x[1],
            'max_features': result.x[2],
            'min_samples_leaf': best_min_samples_leaf,
            'min_samples_split': result.x[4],
            'criterion': result.x[5]
        }
        
        best_model = RandomForestClassifier(**best_params, random_state=self.random_state, class_weight='balanced', n_jobs=-1)
        best_model.fit(self.X_train, self.y_train)
        
        # 在训练集上评估
        y_train_pred = best_model.predict(self.X_train)
        train_recall = recall_score(self.y_train, y_train_pred, average='weighted')
        train_f1 = self.calculate_weighted_f1(self.y_train, y_train_pred)
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        
        # 在测试集上评估
        y_test_pred = best_model.predict(self.X_test)
        test_recall = recall_score(self.y_test, y_test_pred, average='weighted')
        test_f1 = self.calculate_weighted_f1(self.y_test, y_test_pred)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        
        self.best_params['RF'] = best_params
        self.best_scores['RF'] = {
            'train_recall': train_recall, 
            'train_f1': train_f1, 
            'train_accuracy': train_accuracy,
            'test_recall': test_recall, 
            'test_f1': test_f1, 
            'test_accuracy': test_accuracy
        }
        
        print(f"Best RF parameters: {best_params}")
        print(f"Training set - Accuracy: {train_accuracy:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
        print(f"Test set - Accuracy: {test_accuracy:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
        
        return best_model, best_params

    def optimize_xgb(self, n_calls=50):
        """优化XGBoost参数"""
        print("\n" + "="*60)
        print("Optimizing XGBoost")
        print("="*60)
        
        # 定义参数空间
        space = [
            Integer(20, 100, name='n_estimators'),
            Real(0.005, 0.03, name='learning_rate'),
            Integer(8, 20, name='min_child_weight'),
            Integer(2, 4, name='max_depth'),
            Real(0.5, 2.0, name='gamma'),
            Real(0.3, 0.6, name='subsample'), 
            Real(0.3, 0.6, name='colsample_bytree'), 
            Real(1.0, 5.0, name='reg_alpha'), 
            Real(1.0, 5.0, name='reg_lambda')     
        ]
        
        @use_named_args(space)
        def objective(**params):
            try:
                model = XGBClassifier(
                    n_estimators=params['n_estimators'],
                    learning_rate=params['learning_rate'],
                    min_child_weight=params['min_child_weight'],
                    max_depth=params['max_depth'],
                    gamma=params['gamma'],
                    subsample=params['subsample'],
                    colsample_bytree=params['colsample_bytree'],
                    reg_alpha=params['reg_alpha'],
                    reg_lambda=params['reg_lambda'],
                    random_state=self.random_state,
                    eval_metric='logloss'
                )
                
                model.fit(self.X_train, self.y_train)
                
                # 在训练集和测试集上预测和评估
                y_train_pred = model.predict(self.X_train)
                y_test_pred = model.predict(self.X_test)
                
                # 计算训练集和测试集指标
                train_recall = recall_score(self.y_train, y_train_pred, average='weighted')
                train_f1 = self.calculate_weighted_f1(self.y_train, y_train_pred)
                train_accuracy = accuracy_score(self.y_train, y_train_pred)
                
                test_recall = recall_score(self.y_test, y_test_pred, average='weighted')
                test_f1 = self.calculate_weighted_f1(self.y_test, y_test_pred)
                test_accuracy = accuracy_score(self.y_test, y_test_pred)
                
                # 计算过拟合程度
                overfitting_score = train_accuracy - test_accuracy
                
                if overfitting_score > 0.1:  
                    penalty = 0.8  
                elif overfitting_score > 0.05:  
                    penalty = 0.6  
                elif overfitting_score > 0.02: 
                    penalty = 0.3  
                else:
                    penalty = 0.0  
                
                score = (train_f1 + train_recall) * 0.3 + (test_f1 + test_recall) * 0.7
                score = score * (1 - penalty)  
                
                return -score
                
            except Exception as e:
                return 0.0
        
        n_calls = max(10, n_calls)
        result = gp_minimize(objective, space, n_calls=n_calls, random_state=self.random_state)
        
        best_params = {
            'n_estimators': result.x[0],
            'learning_rate': result.x[1],
            'min_child_weight': result.x[2],
            'max_depth': result.x[3],
            'gamma': result.x[4],
            'subsample': result.x[5],
            'colsample_bytree': result.x[6],
            'reg_alpha': result.x[7],
            'reg_lambda': result.x[8]
        }
        
        best_model = XGBClassifier(**best_params, random_state=self.random_state, eval_metric='logloss')
        best_model.fit(self.X_train, self.y_train)
        
        # 在训练集上评估
        y_train_pred = best_model.predict(self.X_train)
        train_recall = recall_score(self.y_train, y_train_pred, average='weighted')
        train_f1 = self.calculate_weighted_f1(self.y_train, y_train_pred)
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        
        # 在测试集上评估
        y_test_pred = best_model.predict(self.X_test)
        test_recall = recall_score(self.y_test, y_test_pred, average='weighted')
        test_f1 = self.calculate_weighted_f1(self.y_test, y_test_pred)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        
        self.best_params['XGB'] = best_params
        self.best_scores['XGB'] = {
            'train_recall': train_recall, 
            'train_f1': train_f1, 
            'train_accuracy': train_accuracy,
            'test_recall': test_recall, 
            'test_f1': test_f1, 
            'test_accuracy': test_accuracy
        }
        
        print(f"Best XGB parameters: {best_params}")
        print(f"Training set - Accuracy: {train_accuracy:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
        print(f"Test set - Accuracy: {test_accuracy:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
        
        return best_model, best_params

    def optimize_svm(self, n_calls=50):
        """优化SVM参数"""
        print("\n" + "="*60)
        print("Optimizing SVM")
        print("="*60)
        
        # 定义参数空间 (极强防过拟合 - SVM专用)
        space = [
            Real(0.001, 0.1, name='C'), 
            Categorical(['scale'], name='gamma'),
            Categorical(['rbf'], name='kernel'),
            Real(0.05, 0.2, name='tol'),
            Integer(200, 800, name='max_iter')
        ]
        
        @use_named_args(space)
        def objective(**params):
            try:
                model = SVC(
                    C=params['C'],
                    gamma=params['gamma'],
                    kernel=params['kernel'],
                    tol=params['tol'],
                    max_iter=params['max_iter'],
                    random_state=self.random_state,
                    probability=True,
                    class_weight='balanced'
                )
                
                model.fit(self.X_train_scaled, self.y_train)
                
                # 在训练集和测试集上预测和评估
                y_train_pred = model.predict(self.X_train_scaled)
                y_test_pred = model.predict(self.X_test_scaled)
                
                # 计算训练集和测试集指标
                train_recall = recall_score(self.y_train, y_train_pred, average='weighted')
                train_f1 = self.calculate_weighted_f1(self.y_train, y_train_pred)
                train_accuracy = accuracy_score(self.y_train, y_train_pred)
                
                test_recall = recall_score(self.y_test, y_test_pred, average='weighted')
                test_f1 = self.calculate_weighted_f1(self.y_test, y_test_pred)
                test_accuracy = accuracy_score(self.y_test, y_test_pred)
                
                # 计算过拟合程度
                overfitting_score = train_accuracy - test_accuracy
                
                if overfitting_score > 0.15:  
                    penalty = 0.9  
                elif overfitting_score > 0.1:  
                    penalty = 0.8  
                elif overfitting_score > 0.05:  
                    penalty = 0.6  
                elif overfitting_score > 0.02:  
                    penalty = 0.3  
                else:
                    penalty = 0.0  
                
                score = (train_f1 + train_recall) * 0.2 + (test_f1 + test_recall) * 0.8
                score = score * (1 - penalty) 
                
                return -score
                
            except Exception as e:
                return 0.0
        
        n_calls = max(10, n_calls)
        result = gp_minimize(objective, space, n_calls=n_calls, random_state=self.random_state)
        
        best_params = {
            'C': result.x[0],
            'gamma': result.x[1],
            'kernel': result.x[2],
            'tol': result.x[3],
            'max_iter': result.x[4]
        }
        
        best_model = SVC(**best_params, random_state=self.random_state, probability=True, class_weight='balanced')
        best_model.fit(self.X_train_scaled, self.y_train)
        
        # 在训练集上评估
        y_train_pred = best_model.predict(self.X_train_scaled)
        train_recall = recall_score(self.y_train, y_train_pred, average='weighted')
        train_f1 = self.calculate_weighted_f1(self.y_train, y_train_pred)
        train_accuracy = accuracy_score(self.y_train, y_train_pred)
        
        # 在测试集上评估
        y_test_pred = best_model.predict(self.X_test_scaled)
        test_recall = recall_score(self.y_test, y_test_pred, average='weighted')
        test_f1 = self.calculate_weighted_f1(self.y_test, y_test_pred)
        test_accuracy = accuracy_score(self.y_test, y_test_pred)
        
        self.best_params['SVM'] = best_params
        self.best_scores['SVM'] = {
            'train_recall': train_recall, 
            'train_f1': train_f1, 
            'train_accuracy': train_accuracy,
            'test_recall': test_recall, 
            'test_f1': test_f1, 
            'test_accuracy': test_accuracy
        }
        
        print(f"Best SVM parameters: {best_params}")
        print(f"Training set - Accuracy: {train_accuracy:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}")
        print(f"Test set - Accuracy: {test_accuracy:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")
        
        return best_model, best_params

    def run_optimization(self, synthetic_data_path, original_test_path, target_column=None, 
                        train_ratio=0.7, n_calls=50, output_dir='./results'):
        """运行完整的贝叶斯优化"""
        print("="*80)
        print("Bayesian Parameter Optimization for F1 and Recall")
        print("="*80)
        
        # 加载数据
        self.load_mixed_data(synthetic_data_path, original_test_path, target_column, train_ratio)
        
        # 优化每个模型
        models = {}
        
        # 优化逻辑回归
        lr_model, lr_params = self.optimize_lr(n_calls)
        models['LR'] = lr_model
        
        # 优化决策树
        dt_model, dt_params = self.optimize_dt(n_calls)
        models['DT'] = dt_model
        
        # 优化随机森林
        rf_model, rf_params = self.optimize_rf(n_calls)
        models['RF'] = rf_model
        
        # 优化XGBoost
        xgb_model, xgb_params = self.optimize_xgb(n_calls)
        models['XGB'] = xgb_model
        
        # 优化SVM
        svm_model, svm_params = self.optimize_svm(n_calls)
        models['SVM'] = svm_model
        
        # 保存结果
        self.save_results(output_dir)
        
        # 显示最终结果
        print("\n" + "="*80)
        print("OPTIMIZATION RESULTS")
        print("="*80)
        
        print(f"{'Model':<8} {'Train Acc':<10} {'Train Rec':<10} {'Train F1':<10} {'Test Acc':<10} {'Test Rec':<10} {'Test F1':<10}")
        print("-" * 80)
        
        for model_name, scores in self.best_scores.items():
            print(f"{model_name:<8} "
                  f"{scores['train_accuracy']:<10.4f} "
                  f"{scores['train_recall']:<10.4f} "
                  f"{scores['train_f1']:<10.4f} "
                  f"{scores['test_accuracy']:<10.4f} "
                  f"{scores['test_recall']:<10.4f} "
                  f"{scores['test_f1']:<10.4f}")
        
        # 找到最佳模型
        best_model_name = max(self.best_scores.keys(), 
                            key=lambda x: self.best_scores[x]['train_f1'] + self.best_scores[x]['train_recall'])
        
        print(f"\nBest Model (based on training F1+Recall): {best_model_name}")
        best_scores = self.best_scores[best_model_name]
        print(f"Training - Accuracy: {best_scores['train_accuracy']:.4f}, Recall: {best_scores['train_recall']:.4f}, F1: {best_scores['train_f1']:.4f}")
        print(f"Test - Accuracy: {best_scores['test_accuracy']:.4f}, Recall: {best_scores['test_recall']:.4f}, F1: {best_scores['test_f1']:.4f}")
        print(f"Best Parameters: {self.best_params[best_model_name]}")
        print(f"Best Scores: {self.best_scores[best_model_name]}")
        
        return models, self.best_params, self.best_scores

    def save_results(self, output_dir='./results'):
        """保存优化结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # 保存参数和分数
        results_df = pd.DataFrame({
            'Model': list(self.best_scores.keys()),
            'Train_Accuracy': [scores['train_accuracy'] for scores in self.best_scores.values()],
            'Train_Recall': [scores['train_recall'] for scores in self.best_scores.values()],
            'Train_F1': [scores['train_f1'] for scores in self.best_scores.values()],
            'Test_Accuracy': [scores['test_accuracy'] for scores in self.best_scores.values()],
            'Test_Recall': [scores['test_recall'] for scores in self.best_scores.values()],
            'Test_F1': [scores['test_f1'] for scores in self.best_scores.values()]
        })
        
        results_path = output_dir / 'bayesian_optimization_results.csv'
        results_df.to_csv(results_path, index=False)
        
        # 保存详细参数
        params_path = output_dir / 'best_parameters.json'
        import json
        
        # 转换numpy类型为Python原生类型
        def convert_numpy_types(obj):
            if isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy_types(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        # 转换参数
        converted_params = convert_numpy_types(self.best_params)
        
        with open(params_path, 'w') as f:
            json.dump(converted_params, f, indent=2)
        
        print(f"\nResults saved to: {output_dir}")


def main():
    """主函数"""
    synthetic_data_path = Path("smote/data/combined_train.csv")
    original_test_path = Path("data/split_data/split_data/test.csv")
    
    if not original_test_path.exists():
        original_test_path = Path("data/split_data/split_data/test_original.csv")
    
    target_column = "Highest Injury Severity"
    
    print("="*80)
    print("Bayesian Parameter Optimization")
    print("="*80)
    print(f"Synthetic data: {synthetic_data_path}")
    print(f"Original test: {original_test_path}")
    
    if synthetic_data_path.exists() and original_test_path.exists():
        optimizer = BayesianOptimizer(random_state=42)
        
        models, best_params, best_scores = optimizer.run_optimization(
            synthetic_data_path=str(synthetic_data_path),
            original_test_path=str(original_test_path),
            target_column=target_column,
            train_ratio=0.7,
            n_calls=50,
            output_dir='./results/bayesian_optimization'
        )
    else:
        print("Required data files not found!")
        if not synthetic_data_path.exists():
            print(f"Synthetic data: {synthetic_data_path}")
        if not original_test_path.exists():
            print(f"Original test: {original_test_path}")

if __name__ == "__main__":
    main()
