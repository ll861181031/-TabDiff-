"""
ADASYN模型评估：边际得分、相关性得分和重复样本比例计算
"""

import pandas as pd
import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import LabelEncoder
import json
from typing import Dict, List, Any


class ADASYNEvaluator:
    """ADASYN模型评估器"""
    
    def __init__(self, real_data_path: str, synthetic_data_path: str, config_path: str):
        """初始化评估器"""
        self.real_data = pd.read_csv(real_data_path)
        self.synthetic_data = pd.read_csv(synthetic_data_path)
        
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        self.column_names = self.config['column_names']
        self.continuous_columns = [self.column_names[idx] for idx in self.config['num_col_idx']]
        self.categorical_columns = [self.column_names[idx] for idx in self.config['cat_col_idx']]
        
        # 编码分类变量
        self._encode_categorical_data()
        
        self.results = {}
    
    def _encode_categorical_data(self):
        """编码分类变量"""
        # 合并数据以统一编码
        all_data = pd.concat([self.real_data, self.synthetic_data], ignore_index=True)
        
        self.encoders = {}
        for col in self.categorical_columns:
            if col in all_data.columns:
                all_data[col] = all_data[col].fillna('Unknown')
                le = LabelEncoder()
                le.fit(all_data[col].astype(str))
                self.encoders[col] = le
        
        # 编码原始数据
        self.real_encoded = self.real_data.copy()
        for col in self.categorical_columns:
            if col in self.real_encoded.columns:
                self.real_encoded[col] = self.real_encoded[col].fillna('Unknown')
                self.real_encoded[col] = self.encoders[col].transform(self.real_encoded[col].astype(str))
        
        # 编码合成数据
        self.synthetic_encoded = self.synthetic_data.copy()
        for col in self.categorical_columns:
            if col in self.synthetic_encoded.columns:
                self.synthetic_encoded[col] = self.synthetic_encoded[col].fillna('Unknown')
                self.synthetic_encoded[col] = self.encoders[col].transform(self.synthetic_encoded[col].astype(str))
        
        # 填充数值列的缺失值
        for col in self.real_encoded.select_dtypes(include=[np.number]).columns:
            self.real_encoded[col] = self.real_encoded[col].fillna(self.real_encoded[col].mean())
            self.synthetic_encoded[col] = self.synthetic_encoded[col].fillna(self.synthetic_encoded[col].mean())
    
    def calculate_marginal_scores(self) -> Dict[str, Any]:
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
        for col in self.continuous_columns:
            if col not in self.real_encoded.columns or col not in self.synthetic_encoded.columns:
                continue
                
            real_vals = self.real_encoded[col].dropna()
            synthetic_vals = self.synthetic_encoded[col].dropna()
            
            if len(real_vals) == 0 or len(synthetic_vals) == 0:
                continue
            
            # 计算KS统计量
            ks_stat, ks_pvalue = stats.ks_2samp(real_vals, synthetic_vals)
            
            # 计算Jensen-Shannon散度
            min_val = min(real_vals.min(), synthetic_vals.min())
            max_val = max(real_vals.max(), synthetic_vals.max())
            bins = np.linspace(min_val, max_val, 11)
            
            real_hist, _ = np.histogram(real_vals, bins=bins, density=True)
            synthetic_hist, _ = np.histogram(synthetic_vals, bins=bins, density=True)
            
            real_hist = real_hist / (real_hist.sum() + 1e-10)
            synthetic_hist = synthetic_hist / (synthetic_hist.sum() + 1e-10)
            
            js_div = jensenshannon(real_hist, synthetic_hist)
            
            # 边际得分 = 1 - JS散度 (越高越好)
            marginal_score = 1 - js_div
            
            marginal_results['continuous_scores'][col] = {
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'js_divergence': js_div,
                'marginal_score': marginal_score
            }
            
            continuous_scores.append(marginal_score)
        
        # 计算分类变量的边际得分
        categorical_scores = []
        for col in self.categorical_columns:
            if col not in self.real_encoded.columns or col not in self.synthetic_encoded.columns:
                continue
                
            real_vals = self.real_encoded[col].dropna()
            synthetic_vals = self.synthetic_encoded[col].dropna()
            
            if len(real_vals) == 0 or len(synthetic_vals) == 0:
                continue
            
            # 计算卡方检验
            real_vals_reset = real_vals.reset_index(drop=True)
            synthetic_vals_reset = synthetic_vals.reset_index(drop=True)
            
            real_labels = pd.Series(['real'] * len(real_vals_reset))
            synthetic_labels = pd.Series(['synthetic'] * len(synthetic_vals_reset))
            
            combined_vals = pd.concat([real_vals_reset, synthetic_vals_reset], ignore_index=True)
            combined_labels = pd.concat([real_labels, synthetic_labels], ignore_index=True)
            
            contingency_table = pd.crosstab(combined_vals, combined_labels)
            
            if contingency_table.shape[0] > 1 and contingency_table.shape[1] > 1:
                chi2_stat, chi2_pvalue, dof, expected = stats.chi2_contingency(contingency_table)
            else:
                chi2_stat, chi2_pvalue = 0, 1
            
            # 计算JS散度
            real_probs = real_vals.value_counts(normalize=True)
            synthetic_probs = synthetic_vals.value_counts(normalize=True)
            
            # 统一索引
            all_categories = set(real_probs.index) | set(synthetic_probs.index)
            real_probs = real_probs.reindex(all_categories, fill_value=0)
            synthetic_probs = synthetic_probs.reindex(all_categories, fill_value=0)
            
            js_div = jensenshannon(real_probs, synthetic_probs)
            marginal_score = 1 - js_div
            
            marginal_results['categorical_scores'][col] = {
                'chi2_statistic': chi2_stat,
                'chi2_pvalue': chi2_pvalue,
                'js_divergence': js_div,
                'marginal_score': marginal_score
            }
            
            categorical_scores.append(marginal_score)
        
        # 计算总体边际得分
        all_scores = continuous_scores + categorical_scores
        if all_scores:
            marginal_results['overall_marginal_score'] = np.mean(all_scores)
        
        # 输出结果
        print(f"连续变量边际得分: {np.mean(continuous_scores):.4f}" if continuous_scores else "无连续变量")
        print(f"分类变量边际得分: {np.mean(categorical_scores):.4f}" if categorical_scores else "无分类变量")
        print(f"总体边际得分: {marginal_results['overall_marginal_score']:.4f}")
        
        # 显示最差的5个变量
        all_scores_dict = {}
        all_scores_dict.update({col: score['marginal_score'] for col, score in marginal_results['continuous_scores'].items()})
        all_scores_dict.update({col: score['marginal_score'] for col, score in marginal_results['categorical_scores'].items()})
        
        if all_scores_dict:
            worst_scores = sorted(all_scores_dict.items(), key=lambda x: x[1])[:5]
            print("\n边际得分最低的5个变量:")
            for col, score in worst_scores:
                print(f"  {col}: {score:.4f}")
        
        self.results['marginal_scores'] = marginal_results
        return marginal_results
    
    def calculate_correlation_scores(self) -> Dict[str, Any]:
        """计算相关性得分 - 评估变量间关系的保持"""
        print("\n" + "="*60)
        print("相关性得分计算 (Correlation Scores)")
        print("="*60)
        
        # 获取所有数值列
        numeric_columns = self.real_encoded.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_columns) < 2:
            print("数值列不足，无法计算相关性得分")
            return {'overall_correlation_score': 0.0}
        
        # 计算相关性矩阵，处理nan值
        real_corr = self.real_encoded[numeric_columns].corr()
        synthetic_corr = self.synthetic_encoded[numeric_columns].corr()
        
        # 计算相关性差异
        corr_diff = np.abs(real_corr - synthetic_corr)
        
        # 计算平均相关性差异，忽略nan值
        mask = np.triu(np.ones_like(corr_diff, dtype=bool), k=1)
        upper_triangle_diff = corr_diff.values[mask]
        # 过滤掉nan值
        valid_diffs = upper_triangle_diff[~np.isnan(upper_triangle_diff)]
        avg_corr_diff = np.mean(valid_diffs) if len(valid_diffs) > 0 else 0
        
        # 相关性得分 = 1 - 平均相关性差异 (越高越好)
        correlation_score = 1 - avg_corr_diff
        
        # 计算互信息得分（用于非线性关系）
        mi_scores = []
        for i, col1 in enumerate(numeric_columns):
            for j, col2 in enumerate(numeric_columns):
                if i < j:  # 避免重复计算
                    try:
                        # 确保数据不为空且有效
                        real_col1 = self.real_encoded[col1].dropna()
                        real_col2 = self.real_encoded[col2].dropna()
                        syn_col1 = self.synthetic_encoded[col1].dropna()
                        syn_col2 = self.synthetic_encoded[col2].dropna()
                        
                        if len(real_col1) > 0 and len(real_col2) > 0 and len(syn_col1) > 0 and len(syn_col2) > 0:
                            real_mi = mutual_info_score(real_col1, real_col2)
                            synthetic_mi = mutual_info_score(syn_col1, syn_col2)
                            
                            if not np.isnan(real_mi) and not np.isnan(synthetic_mi):
                                mi_diff = abs(real_mi - synthetic_mi)
                                mi_scores.append(mi_diff)
                    except:
                        continue
        
        avg_mi_diff = np.mean(mi_scores) if mi_scores else 0
        mi_score = 1 - avg_mi_diff
        
        correlation_results = {
            'correlation_score': correlation_score,
            'mutual_info_score': mi_score,
            'overall_correlation_score': (correlation_score + mi_score) / 2,
            'avg_correlation_diff': avg_corr_diff,
            'avg_mi_diff': avg_mi_diff
        }
        
        print(f"线性相关性得分: {correlation_score:.4f}")
        print(f"互信息得分: {mi_score:.4f}")
        print(f"总体相关性得分: {correlation_results['overall_correlation_score']:.4f}")
        print(f"平均相关性差异: {avg_corr_diff:.4f}")
        
        # 显示相关性变化最大的5对变量
        corr_changes = []
        for i, col1 in enumerate(numeric_columns):
            for j, col2 in enumerate(numeric_columns):
                if i < j:
                    diff = abs(real_corr.loc[col1, col2] - synthetic_corr.loc[col1, col2])
                    corr_changes.append((col1, col2, diff))
        
        corr_changes.sort(key=lambda x: x[2], reverse=True)
        print("\n相关性变化最大的5对变量:")
        for col1, col2, diff in corr_changes[:5]:
            print(f"  {col1} - {col2}: {diff:.4f}")
        
        self.results['correlation_scores'] = correlation_results
        return correlation_results
    
    def calculate_duplicate_ratio(self) -> Dict[str, Any]:
        """计算重复样本比例"""
        print("\n" + "="*60)
        print("重复样本比例计算 (Duplicate Ratio)")
        print("="*60)
        
        # 将数据转换为字符串进行比较
        real_str = self.real_data.astype(str).apply(lambda x: '|'.join(x), axis=1)
        synthetic_str = self.synthetic_data.astype(str).apply(lambda x: '|'.join(x), axis=1)
        
        # 计算完全重复的样本
        exact_duplicates = synthetic_str.isin(real_str).sum()
        exact_duplicate_ratio = exact_duplicates / len(self.synthetic_data)
        
        # 计算部分重复的样本（至少80%的列值相同）
        partial_duplicates = 0
        for i, syn_row in self.synthetic_data.iterrows():
            max_similarity = 0
            for j, real_row in self.real_data.iterrows():
                # 计算行间相似度
                similarity = (syn_row == real_row).sum() / len(syn_row)
                max_similarity = max(max_similarity, similarity)
            
            if max_similarity >= 0.8:  # 80%以上相似度认为是部分重复
                partial_duplicates += 1
        
        partial_duplicate_ratio = partial_duplicates / len(self.synthetic_data)
        
        duplicate_results = {
            'exact_duplicates': exact_duplicates,
            'exact_duplicate_ratio': exact_duplicate_ratio,
            'partial_duplicates': partial_duplicates,
            'partial_duplicate_ratio': partial_duplicate_ratio,
            'unique_samples': len(self.synthetic_data) - exact_duplicates,
            'uniqueness_ratio': 1 - exact_duplicate_ratio
        }
        
        print(f"完全重复样本数: {exact_duplicates}/{len(self.synthetic_data)} ({exact_duplicate_ratio:.2%})")
        print(f"部分重复样本数: {partial_duplicates}/{len(self.synthetic_data)} ({partial_duplicate_ratio:.2%})")
        print(f"独特样本数: {duplicate_results['unique_samples']}/{len(self.synthetic_data)} ({duplicate_results['uniqueness_ratio']:.2%})")
        
        # 评价标准
        if exact_duplicate_ratio < 0.01:
            print("秀：几乎无重复样本")
        elif exact_duplicate_ratio < 0.05:
            print("良好：重复样本很少")
        elif exact_duplicate_ratio < 0.1:
            print("一般：有一定重复样本")
        else:
            print("差：重复样本过多")
        
        self.results['duplicate_ratio'] = duplicate_results
        return duplicate_results
    
    def generate_comprehensive_report(self):
        """生成综合评估报告"""
        print("\n" + "="*80)
        print("ADASYN模型综合评估报告")
        print("="*80)
        
        print(f"\n数据集信息:")
        print(f"  原始数据样本数: {len(self.real_data)}")
        print(f"  合成数据样本数: {len(self.synthetic_data)}")
        print(f"  特征数量: {len(self.column_names)}")
        print(f"  连续变量: {len(self.continuous_columns)}")
        print(f"  分类变量: {len(self.categorical_columns)}")
        
        # 边际得分评价
        if 'marginal_scores' in self.results:
            marginal_score = self.results['marginal_scores']['overall_marginal_score']
            print(f"\n边际得分: {marginal_score:.4f}")
            if marginal_score >= 0.8:
                print("优秀：单变量分布保持良好")
            elif marginal_score >= 0.6:
                print("良好：单变量分布基本保持")
            elif marginal_score >= 0.4:
                print("一般：单变量分布有一定偏差")
            else:
                print("差：单变量分布偏差较大")
        
        # 相关性得分评价
        if 'correlation_scores' in self.results:
            corr_score = self.results['correlation_scores']['overall_correlation_score']
            print(f"\n相关性得分: {corr_score:.4f}")
            if corr_score >= 0.8:
                print("优秀：变量间关系保持良好")
            elif corr_score >= 0.6:
                print("良好：变量间关系基本保持")
            elif corr_score >= 0.4:
                print("一般：变量间关系有一定偏差")
            else:
                print("差：变量间关系偏差较大")
        
        # 重复样本比例评价
        if 'duplicate_ratio' in self.results:
            uniqueness_ratio = self.results['duplicate_ratio']['uniqueness_ratio']
            print(f"\n独特性比例: {uniqueness_ratio:.2%}")
            if uniqueness_ratio >= 0.95:
                print("优秀：合成数据高度独特")
            elif uniqueness_ratio >= 0.9:
                print("良好：合成数据较为独特")
            elif uniqueness_ratio >= 0.8:
                print("一般：合成数据有一定重复")
            else:
                print("差：合成数据重复过多")
        
        # 综合评分
        scores = []
        if 'marginal_scores' in self.results:
            marginal_score = self.results['marginal_scores']['overall_marginal_score']
            if not np.isnan(marginal_score):
                scores.append(marginal_score)
        if 'correlation_scores' in self.results:
            corr_score = self.results['correlation_scores']['overall_correlation_score']
            if not np.isnan(corr_score):
                scores.append(corr_score)
        if 'duplicate_ratio' in self.results:
            uniqueness_score = self.results['duplicate_ratio']['uniqueness_ratio']
            if not np.isnan(uniqueness_score):
                scores.append(uniqueness_score)
        
        if scores:
            overall_score = np.mean(scores)
        else:
            overall_score = 0.0
        
        print(f"\n综合评分: {overall_score:.4f}")
        
        if overall_score >= 0.8:
            print("优秀：ADASYN模型表现优秀")
        elif overall_score >= 0.6:
            print("良好：ADASYN模型表现良好")
        elif overall_score >= 0.4:
            print("一般：ADASYN模型表现一般")
        else:
            print("差：ADASYN模型需要改进")
        
        print("\n" + "="*80)


def main():
    """主函数"""
    print("ADASYN模型评估系统")
    print("="*80)
    
    # 初始化评估器
    evaluator = ADASYNEvaluator(
        real_data_path='data/train.csv',
        synthetic_data_path='data/synthetic_train.csv',
        config_path='data/avoid.json'
    )
    
    # 计算边际得分
    evaluator.calculate_marginal_scores()
    
    # 计算相关性得分
    evaluator.calculate_correlation_scores()
    
    # 计算重复样本比例
    evaluator.calculate_duplicate_ratio()
    
    # 生成综合报告
    evaluator.generate_comprehensive_report()
    
    print("\n评估完成！")


if __name__ == '__main__':
    main()