"""
生成合成数据与原始数据的Spearman相关系数差值热力图
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def load_and_preprocess_data(file_path, model_name):
    """加载和预处理数据"""
    print(f"\n正在处理 {model_name} 数据...")
    
    df = pd.read_csv(file_path)
    print(f"原始数据形状: {df.shape}")
    
    exclude_cols = ['Highest Injury Severity', 'State', 'Make', 'Model Year']
    numeric_cols = []
    
    for col in df.columns:
        if col not in exclude_cols:
            unique_vals = df[col].nunique()
            if unique_vals <= 1:
                print(f"跳过列 '{col}'：只有一个唯一值或全为空值")
                continue
                
            try:
                pd.to_numeric(df[col], errors='raise')
                numeric_cols.append(col)
            except:
                if df[col].dtype == 'object':
                    if df[col].isin(['Y', '', np.nan]).all():
                        df[col] = (df[col] == 'Y').astype(int)
                        numeric_cols.append(col)
                    elif df[col].nunique() < 20:
                        df[col] = pd.Categorical(df[col]).codes
                        numeric_cols.append(col)
    
    print(f"选择的数值型列数量: {len(numeric_cols)}")
    
    df_numeric = df[numeric_cols].copy()
    df_numeric = df_numeric.fillna(df_numeric.median())
    
    return df_numeric, numeric_cols

def create_spearman_difference_heatmap(synthetic_corr, original_corr, output_dir, model_name):
    """创建合成数据与原始数据的Spearman相关系数差值热力图"""
    
    difference_matrix = synthetic_corr - original_corr
    
    plt.figure(figsize=(18, 16))
    plt.gca().set_facecolor('white')
    plt.gcf().patch.set_facecolor('white')
    
    sns.heatmap(difference_matrix, 
                annot=True,
                fmt='.3f',
                cmap='RdBu_r',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                vmin=-0.5, vmax=0.5,
                annot_kws={'size': 8})
    
    plt.title(f'Spearman Correlation Difference: {model_name} vs Original Data', 
              fontsize=18, fontweight='bold', pad=30)
    plt.xlabel('Variables', fontsize=14, fontweight='bold')
    plt.ylabel('Variables', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    heatmap_path = output_dir / f"spearman_correlation_difference_{model_name.lower()}.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"{model_name} Spearman相关系数差值热力图已保存到: {heatmap_path}")
    
    return difference_matrix

def print_difference_summary(difference_matrix, model_name):
    """打印差值分析摘要"""
    print(f"\n{model_name} Spearman Correlation Difference Analysis Summary")
    print("="*60)
    
    diff_values = difference_matrix.values
    np.fill_diagonal(diff_values, np.nan)
    
    max_diff_idx = np.unravel_index(np.nanargmax(diff_values), diff_values.shape)
    max_diff_value = diff_values[max_diff_idx]
    max_diff_vars = (difference_matrix.columns[max_diff_idx[0]], 
                     difference_matrix.columns[max_diff_idx[1]])
    
    min_diff_idx = np.unravel_index(np.nanargmin(diff_values), diff_values.shape)
    min_diff_value = diff_values[min_diff_idx]
    min_diff_vars = (difference_matrix.columns[min_diff_idx[0]], 
                     difference_matrix.columns[min_diff_idx[1]])
    
    print(f"Largest Positive Difference: {max_diff_vars[0]} - {max_diff_vars[1]}: {max_diff_value:.3f}")
    print(f"Largest Negative Difference: {min_diff_vars[0]} - {min_diff_vars[1]}: {min_diff_value:.3f}")
    
    strong_pos = np.sum((diff_values > 0.2) & (diff_values < 1))
    strong_neg = np.sum((diff_values < -0.2) & (diff_values > -1))
    moderate_pos = np.sum((diff_values > 0.1) & (diff_values <= 0.2))
    moderate_neg = np.sum((diff_values < -0.1) & (diff_values >= -0.2))
    weak = np.sum((np.abs(diff_values) <= 0.1) & (diff_values != 0))
    
    print(f"Difference Distribution:")
    print(f"  Strong Positive (>0.2): {strong_pos} pairs")
    print(f"  Moderate Positive (0.1-0.2): {moderate_pos} pairs")
    print(f"  Weak (-0.1 to 0.1): {weak} pairs")
    print(f"  Moderate Negative (-0.2 to -0.1): {moderate_neg} pairs")
    print(f"  Strong Negative (<-0.2): {strong_neg} pairs")
    
    mean_abs_diff = np.nanmean(np.abs(diff_values))
    print(f"Mean Absolute Difference: {mean_abs_diff:.3f}")

def main():
    """主函数"""
    original_data_path = 'data/split_data/split_data/train_original.csv'
    
    synthetic_models = {
        'TabDiff': 'TabDiff/tabdiff/result/avoid/learnable_schedule/1988/samples.csv',
        'ADASYN': 'adasyn/data/combined_train.csv',
        'CTGAN': 'CTGAN/balanced_avoid_data.csv',
        'SMOTE': 'smote/data/combined_train.csv'
    }
    
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("开始为各个合成模型生成Spearman相关系数差值热力图...")
    print("="*80)
    
    print("正在加载原始数据...")
    try:
        original_df, original_numeric_cols = load_and_preprocess_data(original_data_path, "Original")
        original_correlation = original_df.corr(method='spearman')
        print(f"原始数据加载完成，形状: {original_df.shape}")
    except Exception as e:
        print(f"加载原始数据失败: {str(e)}")
        return
    
    all_differences = {}
    
    for model_name, file_path in synthetic_models.items():
        try:
            if not Path(file_path).exists():
                print(f"文件不存在: {file_path}")
                continue
            
            synthetic_df, synthetic_numeric_cols = load_and_preprocess_data(file_path, model_name)
            
            if len(synthetic_numeric_cols) == 0:
                print(f"{model_name} 没有可用的数值型列")
                continue
            
            common_cols = list(set(original_numeric_cols) & set(synthetic_numeric_cols))
            if len(common_cols) == 0:
                print(f"{model_name} 与原始数据没有共同的列")
                continue
            
            print(f"共同列数量: {len(common_cols)}")
            
            original_subset = original_df[common_cols]
            synthetic_subset = synthetic_df[common_cols]
            
            synthetic_correlation = synthetic_subset.corr(method='spearman')
            original_correlation_subset = original_subset.corr(method='spearman')
            
            difference_matrix = create_spearman_difference_heatmap(
                synthetic_correlation, original_correlation_subset, output_dir, model_name)
            
            print_difference_summary(difference_matrix, model_name)
            
            all_differences[model_name] = difference_matrix
            
            print(f"{model_name} 差值热力图生成完成")
            
        except Exception as e:
            print(f"处理 {model_name} 时出错: {str(e)}")
            continue
    
    print("\n" + "="*80)
    print("所有合成模型差值热力图生成完成！")
    print(f"生成的文件保存在: {output_dir}")
    print("\n生成的差值热力图文件:")
    for model_name in synthetic_models.keys():
        if model_name in all_differences:
            print(f"- spearman_correlation_difference_{model_name.lower()}.png")
    
    print("\n所有差值热力图已成功保存到 results/plots/ 文件夹中！")

if __name__ == "__main__":
    main()