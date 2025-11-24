"""
原始训练数据集自变量的Spearman相关系数热力图
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

def load_and_preprocess_data():
    """加载和预处理训练数据"""
    train_path = "data/split_data/split_data/train_original.csv"
    df = pd.read_csv(train_path)
    
    print(f"原始数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
    
    # 选择数值型列进行相关性分析
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
                # 处理分类变量
                if df[col].dtype == 'object':
                    if df[col].isin(['Y', '', np.nan]).all():
                        df[col] = (df[col] == 'Y').astype(int)
                        numeric_cols.append(col)
                    elif df[col].nunique() < 20:
                        df[col] = pd.Categorical(df[col]).codes
                        numeric_cols.append(col)
    
    print(f"选择的数值型列数量: {len(numeric_cols)}")
    print(f"数值型列: {numeric_cols}")
    
    df_numeric = df[numeric_cols].copy()
    df_numeric = df_numeric.fillna(df_numeric.median())
    
    return df_numeric, numeric_cols

def create_spearman_heatmap(df, output_dir):
    """创建Spearman相关系数热力图"""
    correlation_matrix = df.corr(method='spearman')
    
    plt.figure(figsize=(18, 16))
    plt.gca().set_facecolor('white')
    plt.gcf().patch.set_facecolor('white')
    
    sns.heatmap(correlation_matrix, 
                annot=True,
                fmt='.2f',
                cmap='RdBu_r',
                center=0,
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8},
                vmin=-1, vmax=1,
                annot_kws={'size': 8})
    
    plt.title('Spearman Correlation Matrix of Training Dataset Variables', 
              fontsize=18, fontweight='bold', pad=30)
    plt.xlabel('Variables', fontsize=14, fontweight='bold')
    plt.ylabel('Variables', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    heatmap_path = output_dir / "spearman_correlation_heatmap.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"Spearman相关系数热力图已保存到: {heatmap_path}")
    
    return correlation_matrix


def print_correlation_summary(correlation_matrix):
    """打印相关性分析摘要"""
    print("\n" + "="*80)
    print("Spearman Correlation Analysis Summary")
    print("="*80)
    
    # 找出最强的正相关和负相关
    corr_values = correlation_matrix.values
    np.fill_diagonal(corr_values, np.nan)
    
    max_corr_idx = np.unravel_index(np.nanargmax(corr_values), corr_values.shape)
    max_corr_value = corr_values[max_corr_idx]
    max_corr_vars = (correlation_matrix.columns[max_corr_idx[0]], 
                     correlation_matrix.columns[max_corr_idx[1]])
    
    min_corr_idx = np.unravel_index(np.nanargmin(corr_values), corr_values.shape)
    min_corr_value = corr_values[min_corr_idx]
    min_corr_vars = (correlation_matrix.columns[min_corr_idx[0]], 
                     correlation_matrix.columns[min_corr_idx[1]])
    
    print(f"\nStrongest Positive Correlation: {max_corr_vars[0]} - {max_corr_vars[1]}: {max_corr_value:.3f}")
    print(f"Strongest Negative Correlation: {min_corr_vars[0]} - {min_corr_vars[1]}: {min_corr_value:.3f}")
    
    # 统计相关性强度
    strong_pos = np.sum((corr_values > 0.5) & (corr_values < 1))
    strong_neg = np.sum((corr_values < -0.5) & (corr_values > -1))
    moderate_pos = np.sum((corr_values > 0.3) & (corr_values <= 0.5))
    moderate_neg = np.sum((corr_values < -0.3) & (corr_values >= -0.5))
    weak = np.sum((np.abs(corr_values) <= 0.3) & (corr_values != 0))
    
    print(f"\nCorrelation Strength Distribution:")
    print(f"  Strong Positive (>0.5): {strong_pos} pairs")
    print(f"  Moderate Positive (0.3-0.5): {moderate_pos} pairs")
    print(f"  Weak (-0.3 to 0.3): {weak} pairs")
    print(f"  Moderate Negative (-0.5 to -0.3): {moderate_neg} pairs")
    print(f"  Strong Negative (<-0.5): {strong_neg} pairs")
    
    print("\n" + "="*80)

def main():
    """主函数"""
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("正在加载和预处理训练数据...")
    df, numeric_cols = load_and_preprocess_data()
    
    print(f"\n处理后的数据形状: {df.shape}")
    print(f"数值型列数量: {len(numeric_cols)}")
    
    print("\n正在创建Spearman相关系数热力图...")
    correlation_matrix = create_spearman_heatmap(df, output_dir)
    
    print_correlation_summary(correlation_matrix)
    
    print(f"\n热力图已成功保存到 results/plots/ 文件夹中！")
    print(f"生成的文件:")
    print(f"- spearman_correlation_heatmap.png (完整Spearman相关系数热力图)")

if __name__ == "__main__":
    main()
