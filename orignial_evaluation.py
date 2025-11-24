"""
原始数据的分类算法评估与SHAP可解释性分析
包括：LR、DT、RF、XGB、SVM五种分类算法
评估指标：召回率、特异性、加权F1分数、AUC
可解释性：SHAP框架
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
    precision_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report,
    accuracy_score
)
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder

import shap

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class ClassificationEvaluator:
    def __init__(self, random_state=None):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None
        self.scaler = StandardScaler()
        self.all_results = []

        self.models = {
            'LR': LogisticRegression(
                random_state=self.random_state,
                max_iter=3000,
                solver='liblinear',
                C=1.0,
                penalty='l2',
                class_weight=None
            ),
            'DT': DecisionTreeClassifier(
                random_state=self.random_state,
                max_depth=15,
                min_samples_split=15,
                min_samples_leaf=3,
                max_features='log2',
                class_weight=None,
                criterion='gini',
                splitter='best'
            ),
            'RF': RandomForestClassifier(
                n_estimators=40,
                random_state=self.random_state,
                max_depth=8,
                min_samples_split=15,
                min_samples_leaf=12,
                max_features=10,
                class_weight=None,
                bootstrap=True,
                oob_score=True,
                criterion='entropy',
                n_jobs=-1
            ),
            'XGB': XGBClassifier(
                n_estimators=70,
                random_state=self.random_state,
                max_depth=4,
                learning_rate=0.02,
                min_child_weight=15,
                subsample=0.85,
                colsample_bytree=0.9,
                gamma=0.5,
                reg_alpha=0.5,
                reg_lambda=1.5,
                scale_pos_weight=1,
                eval_metric='logloss',
                tree_method='hist',
                grow_policy='lossguide',
                max_leaves=0
            ),
            'SVM': SVC(
                kernel='rbf',
                random_state=self.random_state,
                probability=True,
                gamma='scale',
                C=0.8,
                tol=0.002,
                max_iter=8000,
                class_weight=None,
                decision_function_shape='ovr',
                cache_size=800
            )
        }

    def load_data(self, data_path, target_column=None, test_size=0.2):
        """加载数据集"""
        print(f"Loading data from {data_path}...")

        if str(data_path).endswith('.csv'):
            df = pd.read_csv(data_path)
        else:
            raise ValueError("Only CSV files are supported")

        print(f"Dataset shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")

        if target_column is None:
            target_column = df.columns[-1]

        print(f"Target column: {target_column}")

        X = df.drop(columns=[target_column])
        y = df[target_column]

        # 处理分类特征
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            print(f"Encoding categorical features: {categorical_cols.tolist()}")
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))

        # 编码目标变量
        if y.dtype == 'object' or y.dtype.name == 'category':
            self.label_encoder = LabelEncoder()
            y = self.label_encoder.fit_transform(y)
            print(f"Target classes: {self.label_encoder.classes_}")

        # 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )

        # 标准化特征
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        # 保存特征名
        self.feature_names = X.columns.tolist()

        print(f"Training set: {self.X_train.shape}")
        print(f"Test set: {self.X_test.shape}")
        print(f"Class distribution - Train: {np.bincount(self.y_train)}")
        print(f"Class distribution - Test: {np.bincount(self.y_test)}")

    def load_original_train_test(self, train_data_path, test_data_path, target_column=None):
        """加载原始训练集和测试集进行评估"""
        print(f"Loading original training data from {train_data_path}...")
        print(f"Loading original test data from {test_data_path}...")
        
        if str(train_data_path).endswith('.csv'):
            train_df = pd.read_csv(train_data_path)
        else:
            raise ValueError("Only CSV files are supported for training data")
            
        if str(test_data_path).endswith('.csv'):
            test_df = pd.read_csv(test_data_path)
        else:
            raise ValueError("Only CSV files are supported for test data")

        print(f"Training data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")
        print(f"Columns: {train_df.columns.tolist()}")

        if target_column is None:
            target_column = train_df.columns[-1]

        print(f"Target column: {target_column}")

        X_train = train_df.drop(columns=[target_column])
        y_train = train_df[target_column]
        
        X_test = test_df.drop(columns=[target_column])
        y_test = test_df[target_column]

        # 处理分类特征
        categorical_cols = X_train.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            print(f"Encoding categorical features: {categorical_cols.tolist()}")
            self.label_encoders = {}
            
            for col in categorical_cols:
                le = LabelEncoder()
                all_values = pd.concat([X_train[col], X_test[col]]).astype(str).unique()
                le.fit(all_values)
                
                X_train[col] = le.transform(X_train[col].astype(str))
                X_test[col] = le.transform(X_test[col].astype(str))
                
                self.label_encoders[col] = le
                print(f"  {col}: {len(le.classes_)} unique values")

        # 编码目标变量
        if y_train.dtype == 'object' or y_train.dtype.name == 'category':
            self.label_encoder = LabelEncoder()
            all_target_values = pd.concat([y_train, y_test]).astype(str).unique()
            self.label_encoder.fit(all_target_values)
            
            y_train = self.label_encoder.transform(y_train.astype(str))
            y_test = self.label_encoder.transform(y_test.astype(str))
            print(f"Target classes: {self.label_encoder.classes_}")
            print(f"Target encoding: {dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))}")

        # 设置训练和测试数据
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # 标准化特征
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)

        # 保存特征名
        self.feature_names = X_train.columns.tolist()

        print(f"Training set (original): {self.X_train.shape}")
        print(f"Test set (original): {self.X_test.shape}")
        print(f"Class distribution - Train: {np.bincount(self.y_train)}")
        print(f"Class distribution - Test: {np.bincount(self.y_test)}")
        
        # 检查类别分布差异
        train_classes = np.unique(self.y_train)
        test_classes = np.unique(self.y_test)
        print(f"Training classes: {train_classes}")
        print(f"Test classes: {test_classes}")
        
        if not np.array_equal(train_classes, test_classes):
            print("Warning: Training and test sets have different class distributions!")

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

    def calculate_specificity(self, y_true, y_pred):
        """计算特异性（Specificity）"""
        cm = confusion_matrix(y_true, y_pred)

        # 对于二分类
        if cm.shape[0] == 2:
            tn = cm[0, 0]
            fp = cm[0, 1]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            # 对于多分类，计算宏平均特异性
            specificities = []
            for i in range(cm.shape[0]):
                tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
                fp = cm[:, i].sum() - cm[i, i]
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                specificities.append(spec)
            specificity = np.mean(specificities)

        return specificity

    def calculate_gmean(self, y_true, y_pred):
        """计算G-mean (几何平均)"""
        classes = np.unique(y_true)
        gmeans = []
        
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            tn = np.sum((y_true != cls) & (y_pred != cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            gmean = np.sqrt(recall * specificity) if (recall > 0 and specificity > 0) else 0
            gmeans.append(gmean)
        
        return gmeans

    def calculate_per_class_metrics(self, y_true, y_pred, y_pred_proba):
        """计算每个类别的详细指标"""
        classes = np.unique(y_true)
        class_metrics = {}
        
        for cls in classes:
            tp = np.sum((y_true == cls) & (y_pred == cls))
            tn = np.sum((y_true != cls) & (y_pred != cls))
            fp = np.sum((y_true != cls) & (y_pred == cls))
            fn = np.sum((y_true == cls) & (y_pred != cls))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            gmean = np.sqrt(recall * specificity) if (recall > 0 and specificity > 0) else 0
            
            # 计算AUC
            auc = 0.5
            if len(classes) == 2:
                if cls == 1:
                    try:
                        auc = roc_auc_score((y_true == cls).astype(int), y_pred_proba[:, cls])
                    except:
                        auc = 0.5
                else:
                    try:
                        auc = roc_auc_score((y_true == cls).astype(int), y_pred_proba[:, cls])
                    except:
                        auc = 0.5
            else:
                try:
                    y_binary = (y_true == cls).astype(int)
                    if len(np.unique(y_binary)) > 1:
                        auc = roc_auc_score(y_binary, y_pred_proba[:, cls])
                    else:
                        auc = 0.5
                except:
                    auc = 0.5
            
            class_metrics[cls] = {
                'precision': precision,
                'recall': recall,
                'specificity': specificity,
                'f1': f1,
                'gmean': gmean,
                'auc': auc,
                'support': tp + fn,
                'tp': tp,
                'tn': tn,
                'fp': fp,
                'fn': fn
            }
        
        return class_metrics

    def print_per_class_metrics(self, model_name, class_metrics):
        """打印每个类别的详细指标"""
        print(f"\n=== {model_name} 每个类别的详细指标 ===")
        print(f"{'类别':<6} {'Recall':<8} {'F1':<8} {'G-mean':<8} {'AUC':<8} {'Support':<8}")
        print("-" * 60)
        
        for cls in sorted(class_metrics.keys()):
            metrics = class_metrics[cls]
            print(f"{cls:<6} {metrics['recall']:<8.4f} {metrics['f1']:<8.4f} "
                  f"{metrics['gmean']:<8.4f} {metrics['auc']:<8.4f} {metrics['support']:<8}")
        
        # 计算宏平均
        macro_recall = np.mean([metrics['recall'] for metrics in class_metrics.values()])
        macro_f1 = np.mean([metrics['f1'] for metrics in class_metrics.values()])
        macro_gmean = np.mean([metrics['gmean'] for metrics in class_metrics.values()])
        macro_auc = np.mean([metrics['auc'] for metrics in class_metrics.values()])
        
        print("-" * 60)
        print(f"{'宏平均':<6} {macro_recall:<8.4f} {macro_f1:<8.4f} "
              f"{macro_gmean:<8.4f} {macro_auc:<8.4f}")
        print("=" * 60)

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







    def train_and_evaluate(self):
        """训练和评估所有模型"""
        print("\nTraining and Evaluating Models...")

        for name, model in self.models.items():
            # 训练模型
            if name in ['LR', 'SVM']:
                model.fit(self.X_train_scaled, self.y_train)
                y_pred = model.predict(self.X_test_scaled)
                y_pred_proba = model.predict_proba(self.X_test_scaled)
            else:
                model.fit(self.X_train, self.y_train)
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)

            # 计算评估指标
            recall = recall_score(self.y_test, y_pred, average='weighted')
            specificity = self.calculate_specificity(self.y_test, y_pred)
            f1_weighted = self.calculate_weighted_f1(self.y_test, y_pred)
            accuracy = accuracy_score(self.y_test, y_pred)

            # 计算AUC
            n_classes = len(np.unique(self.y_train))
            if n_classes == 2:
                auc = roc_auc_score(self.y_test, y_pred_proba[:, 1])
            else:
                auc = roc_auc_score(self.y_test, y_pred_proba,
                                   multi_class='ovr', average='weighted')

            # 计算每个类别的详细指标
            class_metrics = self.calculate_per_class_metrics(self.y_test, y_pred, y_pred_proba)

            # 保存结果
            self.results[name] = {
                'model': model,
                'recall': recall,
                'specificity': specificity,
                'f1_weighted': f1_weighted,
                'auc': auc,
                'accuracy': accuracy,
                'y_pred': y_pred,
                'y_pred_proba': y_pred_proba,
                'class_metrics': class_metrics,
                'per_class_metrics': class_metrics
            }

            print(f"{name}: Accuracy={accuracy:.4f}, F1={f1_weighted:.4f}, AUC={auc:.4f}")
            self.print_per_class_metrics(name, class_metrics)

    def find_best_model(self):
        """确定最优模型（基于F1分数）"""
        metrics_df = pd.DataFrame({
            'Model': list(self.results.keys()),
            'Recall': [r['recall'] for r in self.results.values()],
            'Specificity': [r['specificity'] for r in self.results.values()],
            'F1-Weighted': [r['f1_weighted'] for r in self.results.values()],
            'AUC': [r['auc'] for r in self.results.values()],
            'Accuracy': [r['accuracy'] for r in self.results.values()]
        })

        best_idx = metrics_df['F1-Weighted'].idxmax()
        self.best_model_name = metrics_df.loc[best_idx, 'Model']
        self.best_model = self.results[self.best_model_name]['model']

        print(f"\nBest Model: {self.best_model_name} (F1: {metrics_df.loc[best_idx, 'F1-Weighted']:.4f})")

        return metrics_df

    def save_results(self, metrics_df, output_dir='./results'):
        """保存评估结果"""
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

        results_path = output_dir / 'evaluation_results.csv'
        metrics_df.to_csv(results_path, index=False)
        
        self.save_per_class_metrics(output_dir)

        best_model_info = {
            'best_model': self.best_model_name,
            'f1_score': metrics_df.loc[metrics_df['Model'] == self.best_model_name, 'F1-Weighted'].iloc[0],
            'accuracy': metrics_df.loc[metrics_df['Model'] == self.best_model_name, 'Accuracy'].iloc[0],
            'auc': metrics_df.loc[metrics_df['Model'] == self.best_model_name, 'AUC'].iloc[0]
        }

        import json
        with open(output_dir / 'best_model_info.json', 'w') as f:
            json.dump(best_model_info, f, indent=2)

    def save_per_class_metrics(self, output_dir):
        """保存每个类别的详细指标到CSV文件"""
        output_dir = Path(output_dir)
        
        per_class_data = []

        for model_name, result in self.results.items():
            class_metrics = result['class_metrics']

            for class_id, metrics in class_metrics.items():
                per_class_data.append({
                    'Model': model_name,
                    'Class': class_id,
                    'Precision': metrics['precision'],
                    'Recall': metrics['recall'],
                    'F1': metrics['f1'],
                    'G_mean': metrics['gmean'],
                    'AUC': metrics['auc'],
                    'Support': metrics['support']
                })
        
        per_class_df = pd.DataFrame(per_class_data)
        per_class_path = output_dir / 'per_class_metrics.csv'
        per_class_df.to_csv(per_class_path, index=False)
        
        macro_avg_data = []
        for model_name, result in self.results.items():
            class_metrics = result['class_metrics']
            
            macro_recall = np.mean([metrics['recall'] for metrics in class_metrics.values()])
            macro_f1 = np.mean([metrics['f1'] for metrics in class_metrics.values()])
            macro_gmean = np.mean([metrics['gmean'] for metrics in class_metrics.values()])
            macro_auc = np.mean([metrics['auc'] for metrics in class_metrics.values()])
            
            macro_avg_data.append({
                'Model': model_name,
                'Macro_Recall': macro_recall,
                'Macro_F1': macro_f1,
                'Macro_G_mean': macro_gmean,
                'Macro_AUC': macro_auc
            })
        
        macro_avg_df = pd.DataFrame(macro_avg_data)
        macro_avg_path = output_dir / 'macro_average_metrics.csv'
        macro_avg_df.to_csv(macro_avg_path, index=False)
        
        print(f"\n保存结果到:")
        print(f"- 整体指标: {output_dir / 'evaluation_results.csv'}")
        print(f"- 每个类别指标: {output_dir / 'per_class_metrics.csv'}")
        print(f"- 宏平均指标: {output_dir / 'macro_average_metrics.csv'}")

        # 进行SHAP分析
        self.perform_shap_analysis(output_dir)




    def run_mixed_evaluation(self, synthetic_data_path, original_test_path, target_column=None, 
                            train_ratio=0.7, output_dir='./results'):
        """运行混合数据评估流程：合成数据训练，原始测试集评估"""
        self.load_mixed_data(synthetic_data_path, original_test_path, target_column, train_ratio)
        self.train_and_evaluate()
        metrics_df = self.find_best_model()
        self.save_results(metrics_df, output_dir)

        print(f"\nEvaluation completed. Best model: {self.best_model_name}")
        self.create_shap_summary_report(output_dir)

    def perform_shap_analysis(self, output_dir):
        """执行SHAP可解释性分析"""
        print("\n" + "="*80)
        print("SHAP可解释性分析")
        print("="*80)
        
        output_dir = Path(output_dir)
        shap_dir = output_dir / 'shap_analysis'
        shap_dir.mkdir(exist_ok=True)
        
        for model_name, result in self.results.items():
            print(f"\n分析 {model_name} 模型...")
            
            try:
                model = result['model']
                y_pred = result['y_pred']
                y_pred_proba = result['y_pred_proba']
                
                # 创建SHAP解释器
                if model_name == 'LR':
                    explainer = shap.LinearExplainer(model, self.X_train_scaled)
                    shap_values = explainer.shap_values(self.X_test_scaled)
                elif model_name in ['DT', 'RF', 'XGB']:
                    explainer = shap.TreeExplainer(model)
                    shap_values = explainer.shap_values(self.X_test_scaled)
                elif model_name == 'SVM':
                    explainer = shap.KernelExplainer(model.predict_proba, self.X_train_scaled[:50])
                    shap_values = explainer.shap_values(self.X_test_scaled[:30])
                    
                    if isinstance(shap_values, list):
                        for i in range(len(shap_values)):
                            if len(shap_values[i].shape) > 2:
                                shap_values[i] = shap_values[i].reshape(shap_values[i].shape[0], -1)
                    else:
                        if len(shap_values.shape) > 2:
                            shap_values = shap_values.reshape(shap_values.shape[0], -1)
                else:
                    explainer = shap.KernelExplainer(model.predict_proba, self.X_train_scaled[:50])
                    shap_values = explainer.shap_values(self.X_test_scaled[:30])
                
                self._save_shap_values(shap_values, model_name, shap_dir)
                self._create_shap_plots(shap_values, model_name, shap_dir, explainer)
                self._create_feature_importance_analysis(shap_values, model_name, shap_dir)
                self._create_shap_summary_table(shap_values, model_name, shap_dir, y_pred, y_pred_proba)
                self._create_global_feature_importance_plot(shap_values, model_name, shap_dir)
                self._create_shap_summary_beeswarm_plot(shap_values, model_name, shap_dir, explainer)
                
                print(f"{model_name} SHAP分析完成")
                
            except Exception as e:
                print(f"{model_name} SHAP分析失败: {e}")
                continue
        
        print(f"\nSHAP分析完成，结果保存在: {shap_dir}")

    def _save_shap_values(self, shap_values, model_name, shap_dir):
        """保存SHAP值到文件"""
        try:
            if isinstance(shap_values, list):
                for i, class_shap_values in enumerate(shap_values):
                    np.save(shap_dir / f'{model_name}_shap_values_class_{i}.npy', class_shap_values)
                    
                    if len(class_shap_values.shape) == 2:
                        shap_df = pd.DataFrame(class_shap_values, columns=self.feature_names)
                        shap_df.to_csv(shap_dir / f'{model_name}_shap_values_class_{i}.csv', index=False)
            else:
                np.save(shap_dir / f'{model_name}_shap_values.npy', shap_values)
                
                if len(shap_values.shape) == 2:
                    shap_df = pd.DataFrame(shap_values, columns=self.feature_names)
                    shap_df.to_csv(shap_dir / f'{model_name}_shap_values.csv', index=False)
            
            print(f"  - SHAP值已保存 (numpy + CSV): {model_name}")
        except Exception as e:
            print(f"  - 保存SHAP值失败: {e}")

    def _create_shap_plots(self, shap_values, model_name, shap_dir, explainer):
        """创建SHAP可视化图表"""
        try:
            # 创建图片目录
            img_dir = shap_dir / 'img'
            img_dir.mkdir(exist_ok=True)
            
            # 1. 特征重要性条形图
            plt.figure(figsize=(10, 8))
            if isinstance(shap_values, list):
                # 多类别：使用第一个类别的SHAP值
                main_shap_values = shap_values[0]
                # 确保SHAP值和数据形状匹配
                if main_shap_values.shape[1] != self.X_test_scaled.shape[1]:
                    # 截断或填充到匹配长度
                    min_features = min(main_shap_values.shape[1], self.X_test_scaled.shape[1])
                    main_shap_values = main_shap_values[:, :min_features]
                    test_data = self.X_test_scaled[:, :min_features]
                    feature_names = self.feature_names[:min_features]
                else:
                    test_data = self.X_test_scaled
                    feature_names = self.feature_names
                
                shap.summary_plot(main_shap_values, test_data, 
                                feature_names=feature_names, 
                                show=False, plot_type="bar")
            else:
                # 二分类
                # 确保SHAP值和数据形状匹配
                if shap_values.shape[1] != self.X_test_scaled.shape[1]:
                    # 截断或填充到匹配长度
                    min_features = min(shap_values.shape[1], self.X_test_scaled.shape[1])
                    shap_values = shap_values[:, :min_features]
                    test_data = self.X_test_scaled[:, :min_features]
                    feature_names = self.feature_names[:min_features]
                else:
                    test_data = self.X_test_scaled
                    feature_names = self.feature_names
                
                shap.summary_plot(shap_values, test_data, 
                                feature_names=feature_names, 
                                show=False, plot_type="bar")
            
            plt.title(f'{model_name} - Feature Importance (SHAP)')
            plt.tight_layout()
            plt.savefig(img_dir / f'{model_name}_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. SHAP值分布图
            plt.figure(figsize=(12, 8))
            if isinstance(shap_values, list):
                main_shap_values = shap_values[0]
                # 确保SHAP值和数据形状匹配
                if main_shap_values.shape[1] != self.X_test_scaled.shape[1]:
                    min_features = min(main_shap_values.shape[1], self.X_test_scaled.shape[1])
                    main_shap_values = main_shap_values[:, :min_features]
                    test_data = self.X_test_scaled[:, :min_features]
                    feature_names = self.feature_names[:min_features]
                else:
                    test_data = self.X_test_scaled
                    feature_names = self.feature_names
                
                shap.summary_plot(main_shap_values, test_data, 
                                feature_names=feature_names, show=False)
            else:
                # 确保SHAP值和数据形状匹配
                if shap_values.shape[1] != self.X_test_scaled.shape[1]:
                    min_features = min(shap_values.shape[1], self.X_test_scaled.shape[1])
                    shap_values = shap_values[:, :min_features]
                    test_data = self.X_test_scaled[:, :min_features]
                    feature_names = self.feature_names[:min_features]
                else:
                    test_data = self.X_test_scaled
                    feature_names = self.feature_names
                
                shap.summary_plot(shap_values, test_data, 
                                feature_names=feature_names, show=False)
            
            plt.title(f'{model_name} - SHAP Values Distribution')
            plt.tight_layout()
            plt.savefig(img_dir / f'{model_name}_shap_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. 单个样本的SHAP解释（修复waterfall和force_plot问题）
            try:
                # 只处理第一个样本，避免多样本问题
                sample_idx = 0
                
                if isinstance(shap_values, list):
                    # 多类别情况
                    sample_shap_values = shap_values[0][sample_idx]
                    expected_value = explainer.expected_value[0] if hasattr(explainer, 'expected_value') and isinstance(explainer.expected_value, list) else (explainer.expected_value if hasattr(explainer, 'expected_value') else 0)
                else:
                    # 二分类情况
                    sample_shap_values = shap_values[sample_idx]
                    expected_value = explainer.expected_value if hasattr(explainer, 'expected_value') else 0
                
                # 修复形状不匹配问题
                # 确保sample_shap_values是1D数组
                if len(sample_shap_values.shape) > 1:
                    sample_shap_values = sample_shap_values.flatten()
                
                # 确保expected_value是标量
                if isinstance(expected_value, (list, np.ndarray)):
                    expected_value = expected_value[0] if len(expected_value) > 0 else 0
                
                # 确保数据形状匹配
                if len(sample_shap_values) != len(self.X_test_scaled[sample_idx]):
                    # 截断或填充到匹配长度
                    min_len = min(len(sample_shap_values), len(self.X_test_scaled[sample_idx]))
                    sample_shap_values = sample_shap_values[:min_len]
                    sample_data = self.X_test_scaled[sample_idx][:min_len]
                else:
                    sample_data = self.X_test_scaled[sample_idx]
                
                # 尝试使用waterfall_plot
                try:
                    # 修复waterfall_plot需要Explanation对象的问题
                    # 创建Explanation对象
                    if hasattr(shap, 'Explanation'):
                        # 使用新的Explanation对象
                        explanation = shap.Explanation(
                            values=sample_shap_values,
                            base_values=expected_value,
                            data=sample_data,
                            feature_names=self.feature_names[:len(sample_shap_values)]
                        )
                        shap.waterfall_plot(explanation)
                    else:
                        # 使用旧的方法
                        shap.waterfall_plot(expected_value, sample_shap_values, 
                                         sample_data)
                    
                    plt.title(f'{model_name} - Sample {sample_idx+1} SHAP Explanation')
                    plt.tight_layout()
                    plt.savefig(img_dir / f'{model_name}_sample_{sample_idx+1}_explanation.png', 
                               dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"  - 成功生成waterfall图: {model_name}")
                    
                except Exception as waterfall_error:
                    print(f"  - waterfall_plot失败: {waterfall_error}")
                    # 如果waterfall失败，使用force_plot
                    try:
                        # 修复force_plot的base_value参数问题 - 使用新的API
                        if hasattr(shap, 'plots'):
                            # 使用新的shap.plots.force API
                            shap.plots.force(expected_value, sample_shap_values, 
                                           sample_data, 
                                           matplotlib=True)
                        else:
                            # 使用旧的force_plot API - 使用修复后的数据
                            shap.force_plot(expected_value, sample_shap_values, 
                                          sample_data, 
                                          matplotlib=True)
                        plt.title(f'{model_name} - Sample {sample_idx+1} SHAP Explanation')
                        plt.tight_layout()
                        plt.savefig(img_dir / f'{model_name}_sample_{sample_idx+1}_explanation.png', 
                                   dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"  - 成功生成force_plot: {model_name}")
                        
                    except Exception as force_error:
                        print(f"  - force_plot也失败: {force_error}")
                        # 如果都失败，创建一个简单的条形图
                        plt.figure(figsize=(10, 6))
                        feature_names = self.feature_names[:len(sample_shap_values)]
                        plt.barh(range(len(feature_names)), sample_shap_values)
                        plt.yticks(range(len(feature_names)), feature_names)
                        plt.xlabel('SHAP Value')
                        plt.title(f'{model_name} - Sample {sample_idx+1} SHAP Explanation (Simplified)')
                        plt.gca().invert_yaxis()
                        plt.tight_layout()
                        plt.savefig(img_dir / f'{model_name}_sample_{sample_idx+1}_explanation.png', 
                                   dpi=300, bbox_inches='tight')
                        plt.close()
                        print(f"  - 成功生成简化版SHAP解释图: {model_name}")
                
            except Exception as e:
                print(f"  - 创建单个样本解释图失败: {e}")
            
            print(f"  - SHAP图表已保存: {model_name}")
            
        except Exception as e:
            print(f"  - 创建SHAP图表失败: {e}")

    def _create_feature_importance_analysis(self, shap_values, model_name, shap_dir):
        """创建特征重要性分析报告"""
        try:
            # 创建CSV目录
            csv_dir = shap_dir / 'csv'
            csv_dir.mkdir(exist_ok=True)
            
            # 计算特征重要性
            if isinstance(shap_values, list):
                # 多类别：计算所有类别的平均重要性
                mean_shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            else:
                # 二分类
                mean_shap_values = np.abs(shap_values)
            
            # 确保mean_shap_values是2D数组
            if len(mean_shap_values.shape) == 3:
                mean_shap_values = mean_shap_values.reshape(mean_shap_values.shape[0], -1)
            
            # 计算每个特征的平均重要性
            feature_importance = np.mean(mean_shap_values, axis=0)
            
            # 确保特征数量匹配
            if len(feature_importance) != len(self.feature_names):
                print(f"  - 特征数量不匹配: SHAP值{len(feature_importance)} vs 特征名{len(self.feature_names)}")
                # 尝试修复：截断或填充特征重要性
                if len(feature_importance) > len(self.feature_names):
                    feature_importance = feature_importance[:len(self.feature_names)]
                else:
                    # 填充零值
                    padding = np.zeros(len(self.feature_names) - len(feature_importance))
                    feature_importance = np.concatenate([feature_importance, padding])
                print(f"  - 已修复特征数量不匹配问题")
            
            # 确保feature_importance是1D数组
            if len(feature_importance.shape) > 1:
                feature_importance = feature_importance.flatten()
            
            # 创建特征重要性DataFrame
            importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': feature_importance
            }).sort_values('Importance', ascending=False)
            
            # 保存特征重要性到CSV目录
            importance_df.to_csv(csv_dir / f'{model_name}_feature_importance.csv', index=False)
            
            # 创建特征重要性条形图
            img_dir = shap_dir / 'img'
            img_dir.mkdir(exist_ok=True)
            
            plt.figure(figsize=(12, 8))
            top_features = importance_df.head(15)  # 显示前15个重要特征
            
            plt.barh(range(len(top_features)), top_features['Importance'])
            plt.yticks(range(len(top_features)), top_features['Feature'])
            plt.xlabel('SHAP Importance')
            plt.title(f'{model_name} - Top 15 Important Features')
            plt.gca().invert_yaxis()
            plt.tight_layout()
            plt.savefig(img_dir / f'{model_name}_top_features.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"  - 特征重要性分析完成: {model_name}")
            
        except Exception as e:
            print(f"  - 特征重要性分析失败: {e}")

    def _create_shap_summary_table(self, shap_values, model_name, shap_dir, y_pred, y_pred_proba):
        """创建详细的SHAP值汇总表格"""
        try:
            # 创建CSV目录
            csv_dir = shap_dir / 'csv'
            csv_dir.mkdir(exist_ok=True)
            
            # 获取测试样本数量
            n_samples = len(self.X_test_scaled)
            
            # 处理多类别和二分类情况
            if isinstance(shap_values, list):
                # 多类别：使用第一个类别的SHAP值
                main_shap_values = shap_values[0]
            else:
                # 二分类
                main_shap_values = shap_values
            
            # 确保main_shap_values是2D数组
            if len(main_shap_values.shape) == 3:
                main_shap_values = main_shap_values.reshape(main_shap_values.shape[0], -1)
            
            # 确保特征数量匹配
            if main_shap_values.shape[1] != len(self.feature_names):
                print(f"  - 特征数量不匹配: SHAP值{main_shap_values.shape[1]} vs 特征名{len(self.feature_names)}")
                # 尝试修复
                if main_shap_values.shape[1] > len(self.feature_names):
                    main_shap_values = main_shap_values[:, :len(self.feature_names)]
                else:
                    # 填充零值
                    padding = np.zeros((main_shap_values.shape[0], len(self.feature_names) - main_shap_values.shape[1]))
                    main_shap_values = np.concatenate([main_shap_values, padding], axis=1)
                print(f"  - 已修复特征数量不匹配问题")
            
            # 创建汇总表格
            summary_data = []
            
            # 确保样本数量匹配
            actual_samples = min(n_samples, main_shap_values.shape[0])
            
            for i in range(actual_samples):
                # 基本信息
                sample_info = {
                    'Sample_ID': i,
                    'True_Label': self.y_test.iloc[i] if hasattr(self.y_test, 'iloc') else self.y_test[i],
                    'Predicted_Label': y_pred[i],
                    'Prediction_Confidence': np.max(y_pred_proba[i]) if len(y_pred_proba[i]) > 1 else y_pred_proba[i][0]
                }
                
                # 添加原始特征值
                for j, feature_name in enumerate(self.feature_names):
                    sample_info[f'Feature_{feature_name}'] = self.X_test_scaled[i][j]
                
                # 添加SHAP值
                for j, feature_name in enumerate(self.feature_names):
                    if j < main_shap_values.shape[1]:
                        sample_info[f'SHAP_{feature_name}'] = main_shap_values[i][j]
                    else:
                        sample_info[f'SHAP_{feature_name}'] = 0.0
                
                # 计算SHAP值统计
                sample_shap_values = main_shap_values[i]
                sample_info['SHAP_Sum'] = np.sum(sample_shap_values)
                sample_info['SHAP_Max'] = np.max(sample_shap_values)
                sample_info['SHAP_Min'] = np.min(sample_shap_values)
                sample_info['SHAP_Std'] = np.std(sample_shap_values)
                sample_info['SHAP_Abs_Sum'] = np.sum(np.abs(sample_shap_values))
                
                summary_data.append(sample_info)
            
            # 创建DataFrame并保存到CSV目录
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(csv_dir / f'{model_name}_shap_summary_table.csv', index=False)
            
            # 创建特征贡献度表格（每个特征对所有样本的贡献）
            feature_contribution_data = []
            for j, feature_name in enumerate(self.feature_names):
                if j < main_shap_values.shape[1]:
                    feature_shap_values = main_shap_values[:, j]
                    feature_contribution_data.append({
                        'Feature': feature_name,
                        'Mean_SHAP': np.mean(feature_shap_values),
                        'Std_SHAP': np.std(feature_shap_values),
                        'Max_SHAP': np.max(feature_shap_values),
                        'Min_SHAP': np.min(feature_shap_values),
                        'Abs_Mean_SHAP': np.mean(np.abs(feature_shap_values)),
                        'Positive_Count': np.sum(feature_shap_values > 0),
                        'Negative_Count': np.sum(feature_shap_values < 0),
                        'Zero_Count': np.sum(feature_shap_values == 0)
                    })
            
            feature_contribution_df = pd.DataFrame(feature_contribution_data)
            feature_contribution_df = feature_contribution_df.sort_values('Abs_Mean_SHAP', ascending=False)
            feature_contribution_df.to_csv(csv_dir / f'{model_name}_feature_contribution_analysis.csv', index=False)
            
            print(f"  - SHAP汇总表格已保存: {model_name}")
            
        except Exception as e:
            print(f"  - 创建SHAP汇总表格失败: {e}")

    def _create_global_feature_importance_plot(self, shap_values, model_name, shap_dir):
        """创建全局特征重要性排序图 - 显示前20个最重要特征"""
        try:
            # 处理多类别和二分类情况
            if isinstance(shap_values, list):
                # 多类别：计算所有类别的平均重要性
                mean_shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)
            else:
                # 二分类
                mean_shap_values = np.abs(shap_values)
            
            # 确保mean_shap_values是2D数组
            if len(mean_shap_values.shape) == 3:
                mean_shap_values = mean_shap_values.reshape(mean_shap_values.shape[0], -1)
            
            # 计算每个特征的平均重要性
            feature_importance = np.mean(mean_shap_values, axis=0)
            
            # 确保特征数量匹配
            if len(feature_importance) != len(self.feature_names):
                print(f"  - 特征数量不匹配: SHAP值{len(feature_importance)} vs 特征名{len(self.feature_names)}")
                return
            
            # 获取前20个重要特征
            top_20_indices = np.argsort(feature_importance)[-20:][::-1]
            top_20_importance = feature_importance[top_20_indices]
            top_20_features = [self.feature_names[i] for i in top_20_indices]
            
            # 创建图表
            plt.figure(figsize=(12, 10))
            colors = plt.cm.viridis(np.linspace(0, 1, len(top_20_features)))
            
            bars = plt.barh(range(len(top_20_features)), top_20_importance, color=colors)
            plt.yticks(range(len(top_20_features)), top_20_features)
            plt.xlabel('Mean |SHAP Value|', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            plt.title(f'{model_name} - Global Feature Importance Ranking (Top 20 Features)', fontsize=14, fontweight='bold')
            plt.gca().invert_yaxis()
            
            # 添加数值标签
            for i, (bar, importance) in enumerate(zip(bars, top_20_importance)):
                plt.text(importance + 0.001, bar.get_y() + bar.get_height()/2, 
                        f'{importance:.4f}', va='center', ha='left', fontsize=9)
            
            plt.tight_layout()
            plt.savefig(shap_dir / f'{model_name}_global_feature_importance_top20.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # 保存数据到CSV目录
            csv_dir = shap_dir / 'csv'
            csv_dir.mkdir(exist_ok=True)
            
            importance_data = pd.DataFrame({
                'Feature': top_20_features,
                'Importance': top_20_importance,
                'Rank': range(1, len(top_20_features) + 1)
            })
            importance_data.to_csv(csv_dir / f'{model_name}_global_feature_importance_top20.csv', 
                                 index=False)
            
            print(f"  - 全局特征重要性图已保存: {model_name}")
            
        except Exception as e:
            print(f"  - 创建全局特征重要性图失败: {e}")

    def _create_shap_summary_beeswarm_plot(self, shap_values, model_name, shap_dir, explainer):
        """创建SHAP摘要图/Beeswarm图 - 显示特征值对预测的影响"""
        try:
            # 处理多类别和二分类情况
            if isinstance(shap_values, list):
                # 多类别：使用第一个类别的SHAP值
                main_shap_values = shap_values[0]
            else:
                # 二分类
                main_shap_values = shap_values
            
            # 确保main_shap_values是2D数组
            if len(main_shap_values.shape) == 3:
                main_shap_values = main_shap_values.reshape(main_shap_values.shape[0], -1)
            
            # 确保特征数量匹配
            if main_shap_values.shape[1] != len(self.feature_names):
                print(f"  - 特征数量不匹配: SHAP值{main_shap_values.shape[1]} vs 特征名{len(self.feature_names)}")
                return
            
            # 创建SHAP摘要图
            plt.figure(figsize=(12, 10))
            
            # 使用SHAP的summary_plot函数
            shap.summary_plot(main_shap_values, self.X_test_scaled, 
                            feature_names=self.feature_names, 
                            show=False, 
                            max_display=20)  # 显示前20个特征
            
            plt.title(f'{model_name} - SHAP Summary Plot (Beeswarm Plot)', fontsize=14, fontweight='bold')
            plt.xlabel('SHAP Value', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(shap_dir / f'{model_name}_shap_summary_beeswarm.png', 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # 创建详细的SHAP值分析表格
            self._create_shap_beeswarm_analysis_table(main_shap_values, model_name, shap_dir)
            
            print(f"  - SHAP摘要图已保存: {model_name}")
            
        except Exception as e:
            print(f"  - 创建SHAP摘要图失败: {e}")

    def _create_shap_beeswarm_analysis_table(self, shap_values, model_name, shap_dir):
        """创建SHAP Beeswarm分析表格"""
        try:
            # 确保shap_values是2D数组
            if len(shap_values.shape) == 3:
                shap_values = shap_values.reshape(shap_values.shape[0], -1)
            
            # 确保特征数量匹配
            if shap_values.shape[1] != len(self.feature_names):
                print(f"  - 特征数量不匹配: SHAP值{shap_values.shape[1]} vs 特征名{len(self.feature_names)}")
                return
            
            # 计算每个特征的统计信息
            feature_stats = []
            
            for i, feature_name in enumerate(self.feature_names):
                feature_shap_values = shap_values[:, i]
                feature_original_values = self.X_test_scaled[:, i]
                
                # 计算统计信息
                stats = {
                    'Feature': feature_name,
                    'Mean_SHAP': np.mean(feature_shap_values),
                    'Std_SHAP': np.std(feature_shap_values),
                    'Max_SHAP': np.max(feature_shap_values),
                    'Min_SHAP': np.min(feature_shap_values),
                    'Abs_Mean_SHAP': np.mean(np.abs(feature_shap_values)),
                    'Positive_Count': np.sum(feature_shap_values > 0),
                    'Negative_Count': np.sum(feature_shap_values < 0),
                    'Zero_Count': np.sum(feature_shap_values == 0),
                    'Mean_Original_Value': np.mean(feature_original_values),
                    'Std_Original_Value': np.std(feature_original_values),
                    'Correlation_SHAP_Original': np.corrcoef(feature_shap_values, feature_original_values)[0, 1]
                }
                feature_stats.append(stats)
            
            # 创建DataFrame并排序
            stats_df = pd.DataFrame(feature_stats)
            stats_df = stats_df.sort_values('Abs_Mean_SHAP', ascending=False)
            
            # 保存到CSV目录
            csv_dir = shap_dir / 'csv'
            csv_dir.mkdir(exist_ok=True)
            
            stats_df.to_csv(csv_dir / f'{model_name}_shap_beeswarm_analysis.csv', index=False)
            
            # 创建前20个特征的简化表格
            top_20_stats = stats_df.head(20)
            top_20_stats.to_csv(csv_dir / f'{model_name}_shap_beeswarm_top20.csv', index=False)
            
            print(f"  - SHAP Beeswarm分析表格已保存: {model_name}")
            
        except Exception as e:
            print(f"  - 创建SHAP Beeswarm分析表格失败: {e}")

    def create_shap_summary_report(self, output_dir):
        """创建SHAP分析总结报告"""
        output_dir = Path(output_dir)
        shap_dir = output_dir / 'shap_analysis'
        
        if not shap_dir.exists():
            print("SHAP分析目录不存在，跳过总结报告生成")
            return
        
        # 收集所有模型的特征重要性
        all_importance = {}
        
        for model_name in self.results.keys():
            # 检查CSV目录中的特征重要性文件
            csv_dir = shap_dir / 'csv'
            importance_file = csv_dir / f'{model_name}_feature_importance.csv'
            if importance_file.exists():
                try:
                    df = pd.read_csv(importance_file)
                    all_importance[model_name] = df.set_index('Feature')['Importance']
                    print(f"  - 找到特征重要性文件: {model_name}")
                except Exception as e:
                    print(f"  - 读取特征重要性文件失败: {model_name}, {e}")
            else:
                print(f"  - 特征重要性文件不存在: {importance_file}")
        
        if not all_importance:
            print("没有找到特征重要性文件")
            return
        
        # 创建综合特征重要性比较
        importance_comparison = pd.DataFrame(all_importance).fillna(0)
        importance_comparison.to_csv(shap_dir / 'all_models_feature_importance_comparison.csv')
        
        # 创建综合比较图
        plt.figure(figsize=(15, 10))
        importance_comparison.head(15).plot(kind='barh', figsize=(15, 10))
        plt.title('Feature Importance Comparison Across All Models')
        plt.xlabel('SHAP Importance')
        plt.tight_layout()
        plt.savefig(shap_dir / 'all_models_feature_importance_comparison.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"SHAP总结报告已保存到: {shap_dir}")




def main():
    """主函数 - 使用avoid数据集的合成数据进行评估"""
    evaluator = ClassificationEvaluator()

    synthetic_data_path = Path("data/split_data/split_data/train_original.csv")
    
    if not synthetic_data_path.exists():
        print(f"Specified synthetic data not found at: {synthetic_data_path}")
        print("Trying to find alternative samples.csv files...")

    if synthetic_data_path and synthetic_data_path.exists():
        print(f"Using synthetic data: {synthetic_data_path}")
    else:
        print("No synthetic data found!")

    real_test_path = Path("data/split_data/split_data/test.csv")
    target_column = "Highest Injury Severity"

    print("="*80)
    print("Classification Evaluation on Avoid Dataset")
    print("="*80)

    if synthetic_data_path and synthetic_data_path.exists() and real_test_path.exists():
        print(f"\nSynthetic data: {synthetic_data_path}")
        print(f"Original test: {real_test_path}")
        
        evaluator_mixed = ClassificationEvaluator(random_state=42)
        evaluator_mixed.run_mixed_evaluation(
            synthetic_data_path=str(synthetic_data_path),
            original_test_path=str(real_test_path),
            target_column=target_column,
            train_ratio=0.7,
            output_dir='./results/avoid_mixed'
        )
    else:
        print(f"\nRequired data files not found!")
        if not synthetic_data_path or not synthetic_data_path.exists():
            print(f"Synthetic data: {synthetic_data_path}")
        if not real_test_path.exists():
            print(f"Original test: {real_test_path}")


if __name__ == "__main__":
    main()
