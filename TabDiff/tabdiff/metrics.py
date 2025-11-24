from copy import deepcopy
import numpy as np
import torch
import pandas as pd
# Metrics
from eval.mle.mle import get_evaluator
from eval.visualize_density import plot_density
from sdmetrics.reports.single_table import QualityReport, DiagnosticReport
from sdmetrics.single_table import LogisticDetection
from sklearn.preprocessing import OneHotEncoder
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mutual_info_score
from scipy.stats import chi2_contingency, pearsonr, spearmanr

from tqdm import tqdm


class TabMetrics(object):
    def __init__(self, real_data_path, test_data_path, val_data_path, info, device, metric_list) -> None:
        self.real_data_path = real_data_path
        self.test_data_path = test_data_path
        self.val_data_path = val_data_path
        self.info = info
        self.device = device
        self.real_data_size = len(pd.read_csv(real_data_path))
        self.metric_list = metric_list

    def evaluate(self, syn_data):
        metrics, extras = {}, {}
        syn_data_cp = deepcopy(syn_data)
        for metric in self.metric_list:
            func = eval(f"self.evaluate_{metric}")
            print(f"Evaluating {metric}")
            out_metrics, out_extras = func(syn_data_cp)
            metrics.update(out_metrics)
            extras.update(out_extras)
        return metrics, extras
    
    def evaluate_density(self, syn_data):
        real_data = pd.read_csv(self.real_data_path)
        real_data.columns = range(len(real_data.columns))
        syn_data.columns = range(len(syn_data.columns))
        

        info = deepcopy(self.info)
        
        y_only = len(syn_data.columns)==1
        if y_only:
            target_col_idx = info['target_col_idx'][0]
            syn_data = self.complete_y_only_data(syn_data, real_data, target_col_idx)

        metadata = info['metadata']
        metadata['columns'] = {int(key): value for key, value in metadata['columns'].items()} # ensure that keys are all integers?

        new_real_data, new_syn_data, metadata = reorder(real_data, syn_data, info)

        qual_report = QualityReport()
        qual_report.generate(new_real_data, new_syn_data, metadata)

        diag_report = DiagnosticReport()
        diag_report.generate(new_real_data, new_syn_data, metadata)

        quality =  qual_report.get_properties()
        diag = diag_report.get_properties()

        Shape = quality['Score'][0]
        Trend = quality['Score'][1]

        Overall = (Shape + Trend) / 2

        shape_details = qual_report.get_details(property_name='Column Shapes')
        trend_details = qual_report.get_details(property_name='Column Pair Trends')

        if y_only:
            Shape = shape_details['Score'].min()
        out_metrics = {
            "density/Shape": Shape,
            "density/Trend": Trend,
            "density/Overall": Overall,
        }
        out_extras = {
            "shapes": shape_details,
            "trends": trend_details
        }
        return out_metrics, out_extras
    
    def evaluate_mle(self, syn_data):
        train = syn_data.to_numpy()
        test = pd.read_csv(self.test_data_path).to_numpy()
        val = pd.read_csv(self.val_data_path).to_numpy() if self.val_data_path else None
        
        info = deepcopy(self.info)

        task_type = info['task_type']

        evaluator = get_evaluator(task_type)

        if task_type == 'regression':
            best_r2_scores, best_rmse_scores = evaluator(train, test, info, val=val)
            
            overall_scores = {}
            for score_name in ['best_r2_scores', 'best_rmse_scores']:
                overall_scores[score_name] = {}
                
                scores = eval(score_name)
                for method in scores:
                    name = method['name']  
                    method.pop('name')
                    overall_scores[score_name][name] = method 

        else:
            best_f1_scores, best_weighted_scores, best_auroc_scores, best_acc_scores, best_avg_scores = evaluator(train, test, info, val=val)

            overall_scores = {}
            for score_name in ['best_f1_scores', 'best_weighted_scores', 'best_auroc_scores', 'best_acc_scores', 'best_avg_scores']:
                overall_scores[score_name] = {}
                
                scores = eval(score_name)
                for method in scores:
                    name = method['name']  
                    method.pop('name')
                    overall_scores[score_name][name] = method
                    
        if task_type == 'regression':
            mle_score = overall_scores['best_rmse_scores']['XGBRegressor']['RMSE']
        else:
            # 尝试获取XGBClassifier的roc_auc，如果失败则使用其他分类器
            try:
                mle_score = overall_scores['best_auroc_scores']['XGBClassifier']['roc_auc']
                print(f"Debug: XGBClassifier roc_auc = {mle_score}")
                if np.isnan(mle_score):
                    print("Debug: XGBClassifier roc_auc is NaN, trying other classifiers")
                    # 如果XGBClassifier返回NaN，尝试使用其他分类器
                    for classifier_name in overall_scores['best_auroc_scores']:
                        if not np.isnan(overall_scores['best_auroc_scores'][classifier_name]['roc_auc']):
                            mle_score = overall_scores['best_auroc_scores'][classifier_name]['roc_auc']
                            print(f"Debug: Using {classifier_name} roc_auc = {mle_score}")
                            break
            except (KeyError, TypeError) as e:
                print(f"Debug: Error accessing XGBClassifier: {e}")
                # 如果XGBClassifier不存在，使用第一个可用的分类器
                available_classifiers = list(overall_scores['best_auroc_scores'].keys())
                if available_classifiers:
                    mle_score = overall_scores['best_auroc_scores'][available_classifiers[0]]['roc_auc']
                    print(f"Debug: Using first available classifier {available_classifiers[0]} roc_auc = {mle_score}")
                else:
                    mle_score = 0.0
                    print("Debug: No classifiers available, setting mle_score to 0.0")
        
        out_metrics = {
            "mle": mle_score,
        }
        out_extras = {
            "mle": overall_scores,
        }
        return out_metrics, out_extras
    
    def evaluate_c2st(self, syn_data):
        info = deepcopy(self.info)
        real_data = pd.read_csv(self.real_data_path)

        real_data.columns = range(len(real_data.columns))
        syn_data.columns = range(len(syn_data.columns))

        metadata = info['metadata']
        metadata['columns'] = {int(key): value for key, value in metadata['columns'].items()}

        new_real_data, new_syn_data, metadata = reorder(real_data, syn_data, info)

        score = LogisticDetection.compute(
            real_data=new_real_data,
            synthetic_data=new_syn_data,
            metadata=metadata
        )
        
        out_metrics = {
            "c2st": score,
        }
        out_extras = {}
        return out_metrics, out_extras

    def evaluate_dcr(self, syn_data):
        info = deepcopy(self.info)
        real_data = pd.read_csv(self.real_data_path)
        test_data = pd.read_csv(self.test_data_path)
        
        num_col_idx = info['num_col_idx']
        cat_col_idx = info['cat_col_idx']
        target_col_idx = info['target_col_idx']

        task_type = info['task_type']
        if task_type == 'regression':
            num_col_idx += target_col_idx
        else:
            cat_col_idx += target_col_idx

        num_ranges = []

        real_data.columns = list(np.arange(len(real_data.columns)))
        syn_data.columns = list(np.arange(len(real_data.columns)))
        test_data.columns = list(np.arange(len(real_data.columns)))
        for i in num_col_idx:
            num_ranges.append(real_data[i].max() - real_data[i].min()) 
        
        num_ranges = np.array(num_ranges)


        num_real_data = real_data[num_col_idx]
        cat_real_data = real_data[cat_col_idx]
        num_syn_data = syn_data[num_col_idx]
        cat_syn_data = syn_data[cat_col_idx]
        num_test_data = test_data[num_col_idx]
        cat_test_data = test_data[cat_col_idx]

        num_real_data_np = num_real_data.to_numpy()
        cat_real_data_np = cat_real_data.to_numpy().astype('str')
        num_syn_data_np = num_syn_data.to_numpy()
        cat_syn_data_np = cat_syn_data.to_numpy().astype('str')
        num_test_data_np = num_test_data.to_numpy()
        cat_test_data_np = cat_test_data.to_numpy().astype('str')

        encoder = OneHotEncoder()
        cat_complete_data_np = np.concatenate([cat_real_data_np, cat_test_data_np], axis=0)
        encoder.fit(cat_complete_data_np)
        # encoder.fit(cat_real_data_np)


        cat_real_data_oh = encoder.transform(cat_real_data_np).toarray()
        cat_syn_data_oh = encoder.transform(cat_syn_data_np).toarray()
        cat_test_data_oh = encoder.transform(cat_test_data_np).toarray()

        num_real_data_np = num_real_data_np / num_ranges
        num_syn_data_np = num_syn_data_np / num_ranges
        num_test_data_np = num_test_data_np / num_ranges

        real_data_np = np.concatenate([num_real_data_np, cat_real_data_oh], axis=1)
        syn_data_np = np.concatenate([num_syn_data_np, cat_syn_data_oh], axis=1)
        test_data_np = np.concatenate([num_test_data_np, cat_test_data_oh], axis=1)

        device = self.device

        real_data_th = torch.tensor(real_data_np).to(device)
        syn_data_th = torch.tensor(syn_data_np).to(device)  
        test_data_th = torch.tensor(test_data_np).to(device)

        dcrs_real = []
        dcrs_test = []
        batch_size = 10000 // cat_real_data_oh.shape[1]   # This esitmation should make sure that dcr_real and dcr_test can be fit into 10GB GPU memory

        for i in tqdm(range((syn_data_th.shape[0] // batch_size) + 1)):
            if i != (syn_data_th.shape[0] // batch_size):
                batch_syn_data_th = syn_data_th[i*batch_size: (i+1) * batch_size]
            else:
                batch_syn_data_th = syn_data_th[i*batch_size:]
                
            dcr_real = (batch_syn_data_th[:, None] - real_data_th).abs().sum(dim = 2).min(dim = 1).values
            dcr_test = (batch_syn_data_th[:, None] - test_data_th).abs().sum(dim = 2).min(dim = 1).values
            dcrs_real.append(dcr_real)
            dcrs_test.append(dcr_test)
            
        dcrs_real = torch.cat(dcrs_real)
        dcrs_test = torch.cat(dcrs_test)
        
        score = (dcrs_real < dcrs_test).nonzero().shape[0] / dcrs_real.shape[0]
        
        out_metrics = {
            "dcr": score,
        }
        out_extras = {
            "dcr_real": dcrs_real.cpu().numpy(),
            "dcr_test": dcrs_test.cpu().numpy(),
        }
        return out_metrics, out_extras
    
    def evaluate_marginal_score(self, syn_data):
        """
        计算边际得分 - 评估单变量分布的一致性
        使用Jensen-Shannon散度比较真实数据和合成数据的边际分布
        """
        real_data = pd.read_csv(self.real_data_path)
        real_data.columns = range(len(real_data.columns))
        syn_data.columns = range(len(syn_data.columns))
        
        info = deepcopy(self.info)
        num_col_idx = info['num_col_idx']
        cat_col_idx = info['cat_col_idx']
        
        marginal_scores = []
        
        # 评估数值特征的边际分布
        for col_idx in num_col_idx:
            real_col = real_data[col_idx].values
            syn_col = syn_data[col_idx].values
            
            # 移除NaN值
            real_col = real_col[~np.isnan(real_col)]
            syn_col = syn_col[~np.isnan(syn_col)]
            
            if len(real_col) == 0 or len(syn_col) == 0:
                continue
                
            # 创建直方图进行比较
            try:
                # 使用相同的bins
                min_val = min(real_col.min(), syn_col.min())
                max_val = max(real_col.max(), syn_col.max())
                bins = np.linspace(min_val, max_val, 50)
                
                real_hist, _ = np.histogram(real_col, bins=bins, density=True)
                syn_hist, _ = np.histogram(syn_col, bins=bins, density=True)
                
                # 归一化
                real_hist = real_hist / (real_hist.sum() + 1e-8)
                syn_hist = syn_hist / (syn_hist.sum() + 1e-8)
                
                # 计算Jensen-Shannon散度
                js_div = jensenshannon(real_hist, syn_hist)
                marginal_scores.append(1 - js_div)  # 转换为相似度得分
                
            except Exception as e:
                print(f"Warning: Error computing marginal score for column {col_idx}: {e}")
                continue
        
        # 评估分类特征的边际分布
        for col_idx in cat_col_idx:
            real_col = real_data[col_idx].astype(str).values
            syn_col = syn_data[col_idx].astype(str).values
            
            # 获取所有唯一值
            all_values = np.unique(np.concatenate([real_col, syn_col]))
            
            # 计算概率分布
            real_counts = np.array([np.sum(real_col == val) for val in all_values])
            syn_counts = np.array([np.sum(syn_col == val) for val in all_values])
            
            # 归一化
            real_probs = real_counts / (real_counts.sum() + 1e-8)
            syn_probs = syn_counts / (syn_counts.sum() + 1e-8)
            
            # 计算Jensen-Shannon散度
            try:
                js_div = jensenshannon(real_probs, syn_probs)
                marginal_scores.append(1 - js_div)  # 转换为相似度得分
            except Exception as e:
                print(f"Warning: Error computing marginal score for categorical column {col_idx}: {e}")
                continue
        
        # 计算平均边际得分
        if len(marginal_scores) > 0:
            avg_marginal_score = np.mean(marginal_scores)
        else:
            avg_marginal_score = 0.0
        
        out_metrics = {
            "marginal_score": avg_marginal_score,
        }
        out_extras = {
            "marginal_scores_per_column": marginal_scores,
        }
        return out_metrics, out_extras
    
    def evaluate_correlation_score(self, syn_data):
        """
        计算相关性得分 - 评估变量间相关性的保持程度
        比较真实数据和合成数据的相关性矩阵
        """
        real_data = pd.read_csv(self.real_data_path)
        real_data.columns = range(len(real_data.columns))
        syn_data.columns = range(len(syn_data.columns))
        
        info = deepcopy(self.info)
        num_col_idx = info['num_col_idx']
        cat_col_idx = info['cat_col_idx']
        
        correlation_scores = []
        
        # 只对数值特征计算相关性
        if len(num_col_idx) >= 2:
            real_numeric = real_data[num_col_idx]
            syn_numeric = syn_data[num_col_idx]
            
            # 移除包含NaN的行
            real_numeric = real_numeric.dropna()
            syn_numeric = syn_numeric.dropna()
            
            if len(real_numeric) > 1 and len(syn_numeric) > 1:
                try:
                    # 计算Pearson相关系数矩阵
                    real_corr = real_numeric.corr()
                    syn_corr = syn_numeric.corr()
                    
                    # 获取上三角矩阵（避免重复计算）
                    mask = np.triu(np.ones_like(real_corr, dtype=bool), k=1)
                    real_corr_vals = real_corr.values[mask]
                    syn_corr_vals = syn_corr.values[mask]
                    
                    # 移除NaN值
                    valid_mask = ~(np.isnan(real_corr_vals) | np.isnan(syn_corr_vals))
                    real_corr_vals = real_corr_vals[valid_mask]
                    syn_corr_vals = syn_corr_vals[valid_mask]
                    
                    if len(real_corr_vals) > 0:
                        # 计算相关系数的相关性
                        corr_corr, _ = pearsonr(real_corr_vals, syn_corr_vals)
                        if not np.isnan(corr_corr):
                            correlation_scores.append(corr_corr)
                        
                        # 计算均方误差
                        mse = np.mean((real_corr_vals - syn_corr_vals) ** 2)
                        mse_score = 1 / (1 + mse)  # 转换为相似度得分
                        correlation_scores.append(mse_score)
                        
                except Exception as e:
                    print(f"Warning: Error computing correlation score: {e}")
        
        # 计算分类特征的相关性（使用互信息）
        if len(cat_col_idx) >= 2:
            real_categorical = real_data[cat_col_idx].astype(str)
            syn_categorical = syn_data[cat_col_idx].astype(str)
            
            try:
                # 计算真实数据的互信息矩阵
                real_mi_matrix = np.zeros((len(cat_col_idx), len(cat_col_idx)))
                for i, col1 in enumerate(cat_col_idx):
                    for j, col2 in enumerate(cat_col_idx):
                        if i != j:
                            mi = mutual_info_score(real_categorical.iloc[:, i], real_categorical.iloc[:, j])
                            real_mi_matrix[i, j] = mi
                
                # 计算合成数据的互信息矩阵
                syn_mi_matrix = np.zeros((len(cat_col_idx), len(cat_col_idx)))
                for i, col1 in enumerate(cat_col_idx):
                    for j, col2 in enumerate(cat_col_idx):
                        if i != j:
                            mi = mutual_info_score(syn_categorical.iloc[:, i], syn_categorical.iloc[:, j])
                            syn_mi_matrix[i, j] = mi
                
                # 计算互信息矩阵的相关性
                mask = np.triu(np.ones_like(real_mi_matrix, dtype=bool), k=1)
                real_mi_vals = real_mi_matrix[mask]
                syn_mi_vals = syn_mi_matrix[mask]
                
                if len(real_mi_vals) > 0:
                    mi_corr, _ = pearsonr(real_mi_vals, syn_mi_vals)
                    if not np.isnan(mi_corr):
                        correlation_scores.append(mi_corr)
                        
            except Exception as e:
                print(f"Warning: Error computing mutual information correlation: {e}")
        
        # 计算平均相关性得分
        if len(correlation_scores) > 0:
            avg_correlation_score = np.mean(correlation_scores)
        else:
            avg_correlation_score = 0.0
        
        out_metrics = {
            "correlation_score": avg_correlation_score,
        }
        out_extras = {
            "correlation_scores": correlation_scores,
        }
        return out_metrics, out_extras
        
    
    def plot_density(self, syn_data):
        syn_data_cp = deepcopy(syn_data)
        real_data = pd.read_csv(self.real_data_path)
        info = deepcopy(self.info)
        y_only = len(syn_data_cp.columns)==1
        if y_only:
            target_col_idx = info['target_col_idx'][0]
            target_col_name = info['column_names'][target_col_idx]
            syn_data_cp = self.complete_y_only_data(syn_data_cp, real_data, target_col_name)
        img = plot_density(syn_data_cp, real_data, info)
        return img
    
    def complete_y_only_data(self, syn_data, real_data, target_col_idx):
        syn_target_col = deepcopy(syn_data.iloc[:, 0])
        syn_data = deepcopy(real_data)
        syn_data[target_col_idx] = syn_target_col
        return syn_data
        

def reorder(real_data, syn_data, info):
    num_col_idx = deepcopy(info['num_col_idx']) # BUG: info will be modified by += in the next few lines
    cat_col_idx = deepcopy(info['cat_col_idx'])
    target_col_idx = deepcopy(info['target_col_idx'])

    task_type = info['task_type']
    if task_type == 'regression':
        num_col_idx += target_col_idx
    else:
        cat_col_idx += target_col_idx

    real_num_data = real_data[num_col_idx]
    real_cat_data = real_data[cat_col_idx]

    new_real_data = pd.concat([real_num_data, real_cat_data], axis=1)
    new_real_data.columns = range(len(new_real_data.columns))

    syn_num_data = syn_data[num_col_idx]
    syn_cat_data = syn_data[cat_col_idx]
    
    new_syn_data = pd.concat([syn_num_data, syn_cat_data], axis=1)
    new_syn_data.columns = range(len(new_syn_data.columns))

    
    metadata = info['metadata']

    columns = metadata['columns']
    metadata['columns'] = {}

    inverse_idx_mapping = info['inverse_idx_mapping']


    for i in range(len(new_real_data.columns)):
        if i < len(num_col_idx):
            metadata['columns'][i] = columns[num_col_idx[i]]
        else:
            metadata['columns'][i] = columns[cat_col_idx[i-len(num_col_idx)]]
    

    return new_real_data, new_syn_data, metadata