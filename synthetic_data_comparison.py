#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相关性得分和边际分布得分对比图
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def create_synthetic_data_comparison():
    """创建合成数据集对比分析"""
    
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data = {
        'Method': ['TabDiff', 'CTGAN', 'SMOTE', 'ADASYN'],
        'Marginal_Score': [round(0.8992317632154339, 4), round(0.7717, 4), round(0.6538, 4), round(0.7720, 4)],
        'Correlation_Score': [round(0.9171949316804234, 4), round(0.8402, 4), round(0.9492, 4), round(0.8187, 4)]
    }
    
    df = pd.DataFrame(data)
    
    csv_path = output_dir / "synthetic_data_comparison.csv"
    df.to_csv(csv_path, index=False, encoding='utf-8-sig')
    print(f"CSV文件已保存到: {csv_path}")
    
    create_combined_bar_chart(df, output_dir)
    
    return df

def create_combined_bar_chart(df, output_dir):
    """创建综合对比柱状图"""
    x = np.arange(len(df['Method']))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    bars1 = ax.bar(x - width/2, df['Marginal_Score'], width, label='Marginal Score', 
                   color='#1f77b4', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, df['Correlation_Score'], width, label='Correlation Score', 
                   color='#ff7f0e', alpha=0.8, edgecolor='black')
    
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{height:.4f}', ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Generation Method', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Comprehensive Comparison: Marginal Score vs Correlation Score of Different Generation Methods', fontsize=16, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(df['Method'])
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(False)
    
    plt.tight_layout()
    combined_path = output_dir / "synthetic_data_comparison.png"
    plt.savefig(combined_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"综合对比柱状图已保存到: {combined_path}")

def print_summary(df):
    """打印分析摘要"""
    print("\n" + "="*60)
    print("合成数据集生成方法对比分析摘要")
    print("="*60)
    
    print(f"\nMarginal Score Ranking:")
    marginal_ranking = df.sort_values('Marginal_Score', ascending=False)
    for i, (_, row) in enumerate(marginal_ranking.iterrows(), 1):
        print(f"{i}. {row['Method']}: {row['Marginal_Score']:.4f}")
    
    print(f"\nCorrelation Score Ranking:")
    correlation_ranking = df.sort_values('Correlation_Score', ascending=False)
    for i, (_, row) in enumerate(correlation_ranking.iterrows(), 1):
        print(f"{i}. {row['Method']}: {row['Correlation_Score']:.4f}")
    
    df['Comprehensive_Score'] = (df['Marginal_Score'] + df['Correlation_Score']) / 2
    print(f"\nComprehensive Score Ranking (Average of Marginal and Correlation Scores):")
    comprehensive_ranking = df.sort_values('Comprehensive_Score', ascending=False)
    for i, (_, row) in enumerate(comprehensive_ranking.iterrows(), 1):
        print(f"{i}. {row['Method']}: {row['Comprehensive_Score']:.4f}")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    df = create_synthetic_data_comparison()
    print_summary(df)
    
    print(f"\n所有文件已成功保存到 results/plots/ 文件夹中！")
    print(f"生成的文件包括:")
    print(f"- synthetic_data_comparison.csv (数据文件)")
    print(f"- synthetic_data_comparison.png (综合对比柱状图)")