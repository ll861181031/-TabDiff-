"""
SMOTE合成数据的边际得分和相关性得分评估
包含复制数据检测功能
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import LabelEncoder
import json
from typing import Dict, List, Any


class MarginalCorrelationEvaluator:
    """边际得分和相关性得分评估器"""
    
    def __init__(self, real_data: pd.DataFrame, synthetic_data: pd.DataFrame):
        """初始化评估器"""
        self.real_data = real_data
        self.synthetic_data = synthetic_data
        self.results = {}
        
    def calculate_marginal_scores(self, continuous_columns: List[str], 
                                 categorical_columns: List[str]) -> Dict[str, Any]:
        """计算边际得分 - 评估单变量分布的相似性"""
        print("\n" + "="*60)
        print("边际得分计算 (Marginal Scores)")
        print("="*60)
        
        marginal_results = {
            'continuous_scores': {},
            'categorical_scores': {},
            'overall_marginal_score': 0.0
        }
        
        # 计算连续变量的边际得分
        continuous_scores = []
        for col in continuous_columns:
            if col not in self.real_data.columns or col not in self.synthetic_data.columns:
                continue
                
            print(f"\n计算连续变量 '{col}' 的边际得分:")
            
            # 获取数值数据
            real_vals = pd.to_numeric(self.real_data[col], errors='coerce').dropna()
            synthetic_vals = pd.to_numeric(self.synthetic_data[col], errors='coerce').dropna()
            
            if len(real_vals) == 0 or len(synthetic_vals) == 0:
                print(f"  警告: 列 '{col}' 没有有效的数值数据")
                continue
            
            # KS检验
            ks_stat, ks_pvalue = stats.ks_2samp(real_vals, synthetic_vals)
            
            # Wasserstein距离
            wasserstein_dist = stats.wasserstein_distance(real_vals, synthetic_vals)
            
            # Jensen-Shannon散度
            bins = np.linspace(min(real_vals.min(), synthetic_vals.min()), 
                             max(real_vals.max(), synthetic_vals.max()), 50)
            real_hist, _ = np.histogram(real_vals, bins=bins, density=True)
            synthetic_hist, _ = np.histogram(synthetic_vals, bins=bins, density=True)
            
            # 归一化
            real_hist = real_hist / (real_hist.sum() + 1e-10)
            synthetic_hist = synthetic_hist / (synthetic_hist.sum() + 1e-10)
            
            # JS散度计算
            m = (real_hist + synthetic_hist) / 2
            js_divergence = (stats.entropy(real_hist, m) + stats.entropy(synthetic_hist, m)) / 2
            
            # 均值和标准差的相对误差
            mean_error = abs(real_vals.mean() - synthetic_vals.mean()) / (abs(real_vals.mean()) + 1e-10)
            std_error = abs(real_vals.std() - synthetic_vals.std()) / (abs(real_vals.std()) + 1e-10)
            
            # 综合边际得分
            ks_score = max(0, 1 - ks_stat)
            wasserstein_score = max(0, 1 - wasserstein_dist / (real_vals.std() + 1e-10))
            js_score = max(0, 1 - js_divergence)
            mean_score = max(0, 1 - mean_error)
            std_score = max(0, 1 - std_error)
            
            # 加权平均边际得分
            marginal_score = (ks_score * 0.3 + wasserstein_score * 0.3 + 
                            js_score * 0.2 + mean_score * 0.1 + std_score * 0.1)
            
            marginal_results['continuous_scores'][col] = {
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'wasserstein_distance': wasserstein_dist,
                'js_divergence': js_divergence,
                'mean_error': mean_error,
                'std_error': std_error,
                'marginal_score': marginal_score,
                'ks_score': ks_score,
                'wasserstein_score': wasserstein_score,
                'js_score': js_score,
                'mean_score': mean_score,
                'std_score': std_score
            }
            
            continuous_scores.append(marginal_score)
            
            print(f"  KS统计量: {ks_stat:.4f} (p值: {ks_pvalue:.4f})")
            print(f"  Wasserstein距离: {wasserstein_dist:.4f}")
            print(f"  JS散度: {js_divergence:.4f}")
            print(f"  均值相对误差: {mean_error:.4f}")
            print(f"  标准差相对误差: {std_error:.4f}")
            print(f"  边际得分: {marginal_score:.4f}")
        
        # 计算分类变量的边际得分
        categorical_scores = []
        for col in categorical_columns:
            if col not in self.real_data.columns or col not in self.synthetic_data.columns:
                continue
                
            print(f"\n计算分类变量 '{col}' 的边际得分:")
            
            # 获取分类数据
            real_vals = self.real_data[col].astype(str)
            synthetic_vals = self.synthetic_data[col].astype(str)
            
            # 计算分布
            real_dist = real_vals.value_counts(normalize=True).sort_index()
            synthetic_dist = synthetic_vals.value_counts(normalize=True).sort_index()
            
            # Jensen-Shannon散度
            all_categories = set(real_dist.index) | set(synthetic_dist.index)
            real_probs = np.array([real_dist.get(cat, 0) for cat in all_categories])
            synthetic_probs = np.array([synthetic_dist.get(cat, 0) for cat in all_categories])
            
            # 避免零概率
            real_probs = real_probs + 1e-10
            synthetic_probs = synthetic_probs + 1e-10
            real_probs = real_probs / real_probs.sum()
            synthetic_probs = synthetic_probs / synthetic_probs.sum()
            
            m = (real_probs + synthetic_probs) / 2
            js_divergence = (stats.entropy(real_probs, m) + stats.entropy(synthetic_probs, m)) / 2
            
            # 卡方检验
            all_cats = sorted(all_categories)
            real_counts = [real_vals.value_counts().get(cat, 0) for cat in all_cats]
            synthetic_counts = [synthetic_vals.value_counts().get(cat, 0) for cat in all_cats]
            
            if len(all_cats) > 1 and sum(real_counts) > 0 and sum(synthetic_counts) > 0:
                contingency_table = np.array([real_counts, synthetic_counts])
                chi2_stat, chi2_pvalue, dof, expected = stats.chi2_contingency(contingency_table)
            else:
                chi2_stat, chi2_pvalue = 0, 1
            
            # 互信息
            le = LabelEncoder()
            all_vals = pd.concat([real_vals, synthetic_vals])
            le.fit(all_vals)
            real_encoded = le.transform(real_vals)
            synthetic_encoded = le.transform(synthetic_vals)
            
            # 计算互信息
            min_len = min(len(real_encoded), len(synthetic_encoded))
            if min_len > 0:
                real_sample = np.random.choice(real_encoded, min_len, replace=False)
                synthetic_sample = np.random.choice(synthetic_encoded, min_len, replace=False)
                mi_score = mutual_info_score(real_sample, synthetic_sample)
            else:
                mi_score = 0
            
            # 综合边际得分
            js_score = max(0, 1 - js_divergence)
            chi2_score = max(0, 1 - chi2_stat / (len(all_cats) + 1e-10))
            mi_score_norm = min(1, mi_score)
            
            marginal_score = (js_score * 0.5 + chi2_score * 0.3 + mi_score_norm * 0.2)
            
            marginal_results['categorical_scores'][col] = {
                'js_divergence': js_divergence,
                'chi2_statistic': chi2_stat,
                'chi2_pvalue': chi2_pvalue,
                'mutual_info': mi_score,
                'marginal_score': marginal_score,
                'js_score': js_score,
                'chi2_score': chi2_score,
                'mi_score': mi_score_norm
            }
            
            categorical_scores.append(marginal_score)
            
            print(f"  JS散度: {js_divergence:.4f}")
            print(f"  卡方统计量: {chi2_stat:.4f} (p值: {chi2_pvalue:.4f})")
            print(f"  互信息: {mi_score:.4f}")
            print(f"  边际得分: {marginal_score:.4f}")
        
        # 计算总体边际得分
        all_scores = continuous_scores + categorical_scores
        if all_scores:
            marginal_results['overall_marginal_score'] = np.mean(all_scores)
            print(f"\n总体边际得分: {marginal_results['overall_marginal_score']:.4f}")
        
        return marginal_results
    
    def calculate_correlation_scores(self, continuous_columns: List[str]) -> Dict[str, Any]:
        """计算相关性得分 - 评估变量间相关性的保持程度"""
        print("\n" + "="*60)
        print("相关性得分计算 (Correlation Scores)")
        print("="*60)
        
        correlation_results = {
            'pearson_correlation': {},
            'spearman_correlation': {},
            'overall_correlation_score': 0.0
        }
        
        # 确保所有列都存在
        valid_columns = [col for col in continuous_columns 
                        if col in self.real_data.columns and col in self.synthetic_data.columns]
        
        if len(valid_columns) < 2:
            print("警告: 连续变量数量不足，无法计算相关性")
            return correlation_results
        
        # 获取数值数据
        real_numeric = self.real_data[valid_columns].apply(pd.to_numeric, errors='coerce')
        synthetic_numeric = self.synthetic_data[valid_columns].apply(pd.to_numeric, errors='coerce')
        
        # 删除包含NaN的行
        real_numeric = real_numeric.dropna()
        synthetic_numeric = synthetic_numeric.dropna()
        
        if len(real_numeric) == 0 or len(synthetic_numeric) == 0:
            print("警告: 没有有效的数值数据计算相关性")
            return correlation_results
        
        # Pearson相关系数矩阵
        real_pearson = real_numeric.corr()
        synthetic_pearson = synthetic_numeric.corr()
        
        # 计算Pearson相关性差异
        pearson_diff = np.abs(real_pearson - synthetic_pearson)
        pearson_mean_diff = pearson_diff.mean().mean()
        pearson_max_diff = pearson_diff.max().max()
        
        # Pearson相关性得分
        pearson_score = max(0, 1 - pearson_mean_diff)
        
        correlation_results['pearson_correlation'] = {
            'mean_difference': pearson_mean_diff,
            'max_difference': pearson_max_diff,
            'correlation_score': pearson_score,
            'real_correlation_matrix': real_pearson,
            'synthetic_correlation_matrix': synthetic_pearson
        }
        
        print(f"Pearson相关系数分析:")
        print(f"  平均差异: {pearson_mean_diff:.4f}")
        print(f"  最大差异: {pearson_max_diff:.4f}")
        print(f"  相关性得分: {pearson_score:.4f}")
        
        # Spearman等级相关系数矩阵
        real_spearman = real_numeric.corr(method='spearman')
        synthetic_spearman = synthetic_numeric.corr(method='spearman')
        
        # 计算Spearman相关性差异
        spearman_diff = np.abs(real_spearman - synthetic_spearman)
        spearman_mean_diff = spearman_diff.mean().mean()
        spearman_max_diff = spearman_diff.max().max()
        
        # Spearman相关性得分
        spearman_score = max(0, 1 - spearman_mean_diff)
        
        correlation_results['spearman_correlation'] = {
            'mean_difference': spearman_mean_diff,
            'max_difference': spearman_max_diff,
            'correlation_score': spearman_score,
            'real_correlation_matrix': real_spearman,
            'synthetic_correlation_matrix': synthetic_spearman
        }
        
        print(f"\nSpearman等级相关系数分析:")
        print(f"  平均差异: {spearman_mean_diff:.4f}")
        print(f"  最大差异: {spearman_max_diff:.4f}")
        print(f"  相关性得分: {spearman_score:.4f}")
        
        # 计算总体相关性得分
        correlation_results['overall_correlation_score'] = (pearson_score + spearman_score) / 2
        print(f"\n总体相关性得分: {correlation_results['overall_correlation_score']:.4f}")
        
        return correlation_results
    
    def detect_duplicate_data(self) -> Dict[str, Any]:
        """检测合成数据集中是否有从原始数据集复制的数据"""
        print("\n" + "="*60)
        print("复制数据检测 (Duplicate Data Detection)")
        print("="*60)
        
        # 将数据转换为字符串进行比较
        real_str = self.real_data.astype(str).apply(lambda x: '|'.join(x), axis=1)
        synthetic_str = self.synthetic_data.astype(str).apply(lambda x: '|'.join(x), axis=1)
        
        # 检查完全相同的样本
        exact_duplicates = synthetic_str.isin(real_str).sum()
        exact_duplicate_rate = exact_duplicates / len(self.synthetic_data) * 100
        
        # 检查部分匹配（至少80%的列值相同）
        partial_duplicates = 0
        for i, syn_row in enumerate(self.synthetic_data.iterrows()):
            syn_vals = syn_row[1].astype(str).values
            for j, real_row in enumerate(self.real_data.iterrows()):
                real_vals = real_row[1].astype(str).values
                # 计算相同值的比例
                same_vals = np.sum(syn_vals == real_vals)
                similarity = same_vals / len(syn_vals)
                if similarity >= 0.8:  # 80%以上相似
                    partial_duplicates += 1
                    break
        
        partial_duplicate_rate = partial_duplicates / len(self.synthetic_data) * 100
        
        # 计算数据质量评分
        if exact_duplicate_rate < 1:
            quality_rating = "优秀"
        elif exact_duplicate_rate < 5:
            quality_rating = "良好"
        elif exact_duplicate_rate < 10:
            quality_rating = "一般"
        else:
            quality_rating = "差"
        
        results = {
            'exact_duplicates': exact_duplicates,
            'exact_duplicate_rate': exact_duplicate_rate,
            'partial_duplicates': partial_duplicates,
            'partial_duplicate_rate': partial_duplicate_rate,
            'quality_rating': quality_rating,
            'total_synthetic_samples': len(self.synthetic_data),
            'total_real_samples': len(self.real_data)
        }
        
        print(f"合成数据总样本数: {len(self.synthetic_data)}")
        print(f"原始数据总样本数: {len(self.real_data)}")
        print(f"\n完全相同的样本: {exact_duplicates}/{len(self.synthetic_data)} ({exact_duplicate_rate:.2f}%)")
        print(f"高度相似的样本 (≥80%): {partial_duplicates}/{len(self.synthetic_data)} ({partial_duplicate_rate:.2f}%)")
        print(f"\n数据质量评级: {quality_rating}")
        
        if exact_duplicate_rate >= 10:
            print("\n警告: 复制数据比例过高，SMOTE可能退化为简单复制")
        
        return results
    
    def evaluate_all(self, continuous_columns: List[str], categorical_columns: List[str]) -> Dict[str, Any]:
        """执行完整的评估流程"""
        print("开始SMOTE合成数据质量评估...")
        print("="*60)
        
        # 边际得分计算
        marginal_results = self.calculate_marginal_scores(continuous_columns, categorical_columns)
        
        # 相关性得分计算
        correlation_results = self.calculate_correlation_scores(continuous_columns)
        
        # 复制数据检测
        duplicate_results = self.detect_duplicate_data()
        
        # 整合所有结果
        all_results = {
            'marginal_scores': marginal_results,
            'correlation_scores': correlation_results,
            'duplicate_detection': duplicate_results,
            'summary': {
                'overall_marginal_score': marginal_results['overall_marginal_score'],
                'overall_correlation_score': correlation_results['overall_correlation_score'],
                'exact_duplicate_rate': duplicate_results['exact_duplicate_rate'],
                'quality_rating': duplicate_results['quality_rating']
            }
        }
        
        # 打印总结
        print("\n" + "="*60)
        print("评估结果总结")
        print("="*60)
        print(f"总体边际得分: {marginal_results['overall_marginal_score']:.4f}")
        print(f"总体相关性得分: {correlation_results['overall_correlation_score']:.4f}")
        print(f"复制数据比例: {duplicate_results['exact_duplicate_rate']:.2f}%")
        print(f"数据质量评级: {duplicate_results['quality_rating']}")
        
        return all_results


def load_config(config_path='data/avoid.json'):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return json.load(f)


def main():
    """主函数"""
    print("SMOTE合成数据边际得分和相关性得分评估")
    print("="*60)
    
    # 加载配置
    config = load_config('data/avoid.json')
    
    # 加载数据
    print("加载数据...")
    real_data = pd.read_csv('data/train.csv')
    synthetic_data = pd.read_csv('data/synthetic_train.csv')
    
    print(f"原始数据: {len(real_data)} 样本")
    print(f"合成数据: {len(synthetic_data)} 样本")
    
    # 获取列信息
    column_names = config['column_names']
    continuous_columns = [column_names[idx] for idx in config['num_col_idx']]
    categorical_columns = [column_names[idx] for idx in config['cat_col_idx']]
    
    print(f"\n连续变量: {len(continuous_columns)} 个")
    print(f"分类变量: {len(categorical_columns)} 个")
    
    # 创建评估器
    evaluator = MarginalCorrelationEvaluator(real_data, synthetic_data)
    
    # 执行评估
    results = evaluator.evaluate_all(continuous_columns, categorical_columns)
    


if __name__ == '__main__':
    main()