"""
性能指标对比图
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

def load_model_data():
    """从5个CSV汇总文件中加载数据"""
    algorithms = ['LR', 'DT', 'RF', 'XGB', 'SVM']
    all_data = []
    
    for algorithm in algorithms:
        csv_path = f"results/plots/{algorithm.lower()}_summary.csv"
        try:
            df_alg = pd.read_csv(csv_path)
            print(f"成功加载{algorithm}数据: {csv_path}")
            
            datasets = ['Original', 'Smote', 'CtGan', 'AdaSyn', 'TabDiff']
            
            for dataset in datasets:
                row_data = {
                    'Original': dataset,
                    'Model': algorithm,
                    'Total Accuracy': 0.0,
                    'Accuracy_class 0': 0.0,
                    'Accuracy_class 1': 0.0,
                    'Accuracy_class 2': 0.0,
                    'Accuracy_class 4': 0.0,
                    'F1-Score': 0.0,
                    'G-mean': 0.0,
                    'AUC': 0.0
                }
                
                for _, csv_row in df_alg.iterrows():
                    metric_name = csv_row['Metric']
                    if metric_name in row_data:
                        row_data[metric_name] = csv_row[dataset]
                
                all_data.append(row_data)
                
        except FileNotFoundError:
            print(f"警告: 未找到文件 {csv_path}")
            continue
    
    df = pd.DataFrame(all_data)
    
    print(f"数据形状: {df.shape}")
    print(f"数据集: {df['Original'].unique()}")
    print(f"算法: {df['Model'].unique()}")
    print(f"指标: {[col for col in df.columns if col not in ['Original', 'Model']]}")
    
    return df


def create_individual_algorithm_charts(df, output_dir):
    """为每个算法创建单独的综合性能对比图"""
    algorithms = ['LR', 'DT', 'RF', 'XGB', 'SVM']
    metrics = ['Total Accuracy', 'Accuracy_class 0', 'Accuracy_class 1', 'Accuracy_class 2', 'Accuracy_class 4', 'F1-Score', 'G-mean', 'AUC']
    datasets = df['Original'].unique()
    
    colors = ['#87CEEB', '#FFA500', '#808080', '#FFFF00', '#0000FF']
    
    for algorithm in algorithms:
        plt.figure(figsize=(14, 8))
        plt.gca().set_facecolor('white')
        plt.gcf().patch.set_facecolor('white')
        
        alg_data = df[df['Model'] == algorithm]
        
        x = np.arange(len(metrics))
        width = 0.15
        
        for i, dataset in enumerate(datasets):
            dataset_data = alg_data[alg_data['Original'] == dataset]
            if len(dataset_data) > 0:
                values = [dataset_data[metric].iloc[0] for metric in metrics]
                plt.bar(x + i * width, values, width, 
                       label=dataset, color=colors[i], alpha=0.8, edgecolor='black')
        
        plt.xlabel('Performance Metrics', fontsize=14, fontweight='bold')
        plt.ylabel('Performance Score', fontsize=14, fontweight='bold')
        plt.title(f'{algorithm} Algorithm Performance Comparison Across Different Data Augmentation Methods', 
                 fontsize=16, fontweight='bold', pad=20)
        plt.xticks(x + width * 2, metrics, rotation=0)
        plt.legend(loc='upper left', fontsize=12)
        plt.ylim(0, 1.0)
        plt.grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for i, dataset in enumerate(datasets):
            dataset_data = alg_data[alg_data['Original'] == dataset]
            if len(dataset_data) > 0:
                for j, metric in enumerate(metrics):
                    value = dataset_data[metric].iloc[0]
                    plt.text(j + i * width, value + 0.01, f'{value:.3f}', 
                            ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        chart_path = output_dir / f"{algorithm.lower()}_comprehensive_performance.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"{algorithm}综合性能对比图已保存到: {chart_path}")

def create_summary_chart(df, output_dir):
    """所有算法的汇总对比图"""
    algorithms = ['LR', 'DT', 'RF', 'XGB', 'SVM']
    metrics = ['Total Accuracy', 'Accuracy_class 0', 'Accuracy_class 1', 'Accuracy_class 2', 'Accuracy_class 4', 'F1-Score', 'G-mean', 'AUC']
    datasets = df['Original'].unique()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    plt.figure(figsize=(22, 20))
    plt.gca().set_facecolor('white')
    plt.gcf().patch.set_facecolor('white')
    
    y_positions = []
    y_labels = []
    bar_data = []
    
    metric_spacing = 10
    algorithm_spacing = 2.5
    
    for i, metric in enumerate(metrics):
        for j, algorithm in enumerate(algorithms):
            for k, dataset in enumerate(datasets):
                alg_data = df[(df['Model'] == algorithm) & (df['Original'] == dataset)]
                if len(alg_data) > 0:
                    value = alg_data[metric].iloc[0]
                    y_pos = i * metric_spacing + j * algorithm_spacing + k * 0.3
                    y_positions.append(y_pos)
                    y_labels.append(f'{metric}\n{algorithm}\n{dataset}')
                    bar_data.append((value, colors[k]))
    
    y_pos = np.array(y_positions)
    values = [item[0] for item in bar_data]
    bar_colors = [item[1] for item in bar_data]
    
    bars = plt.barh(y_pos, values, height=0.25, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    
    simplified_labels = []
    for i, metric in enumerate(metrics):
        for j, algorithm in enumerate(algorithms):
            for k, dataset in enumerate(datasets):
                alg_data = df[(df['Model'] == algorithm) & (df['Original'] == dataset)]
                if len(alg_data) > 0:
                    if k == 0:
                        simplified_labels.append(f'{metric}\n{algorithm}')
                    else:
                        simplified_labels.append('')
    
    plt.yticks(y_pos, simplified_labels, fontsize=10)
    
    # 添加数值标签
    for i, (bar, value) in enumerate(zip(bars, values)):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{value:.3f}', ha='left', va='center', fontsize=8, fontweight='bold')
    
    plt.xlabel('Performance Score', fontsize=16, fontweight='bold')
    plt.title('All Algorithms Performance Comparison Across Different Data Augmentation Methods', 
             fontsize=18, fontweight='bold', pad=30)
    plt.xlim(0, 1.0)
    plt.grid(True, alpha=0.3, axis='x')
    
    # 添加图例
    legend_elements = [plt.Rectangle((0,0),1,1, color=colors[i], alpha=0.8, label=dataset) 
                      for i, dataset in enumerate(datasets)]
    plt.legend(handles=legend_elements, loc='lower right', fontsize=14)
    
    plt.ylim(min(y_pos) - 1, max(y_pos) + 1)
    plt.tight_layout()
    
    summary_path = output_dir / "all_algorithms_summary_comparison.png"
    plt.savefig(summary_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"所有算法汇总对比图已保存到: {summary_path}")


def print_performance_summary(df):
    """打印性能摘要"""
    print("\n" + "="*80)
    print("Algorithm Performance Summary")
    print("="*80)
    
    algorithms = ['LR', 'DT', 'RF', 'XGB', 'SVM']
    metrics = ['Total Accuracy', 'Accuracy_class 0', 'Accuracy_class 1', 'Accuracy_class 2', 'Accuracy_class 4', 'F1-Score', 'G-mean', 'AUC']
    
    for algorithm in algorithms:
        print(f"\n{algorithm} Algorithm:")
        alg_data = df[df['Model'] == algorithm]
        for metric in metrics:
            avg_score = alg_data[metric].mean()
            std_score = alg_data[metric].std()
            print(f"  {metric}: {avg_score:.3f} ± {std_score:.3f}")
    
    print("\n" + "="*80)

def main():
    """主函数"""
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("正在加载模型性能数据...")
    df = load_model_data()
    
    print("\n正在创建各算法单独性能对比图...")
    create_individual_algorithm_charts(df, output_dir)
    
    print("\n正在创建所有算法汇总对比图...")
    create_summary_chart(df, output_dir)
    
    print_performance_summary(df)
    
    print(f"\n所有图表已成功保存到 results/plots/ 文件夹中！")
    print(f"\n生成的文件包括:")
    print(f"- 各算法综合性能对比图: lr_comprehensive_performance.png, dt_comprehensive_performance.png, rf_comprehensive_performance.png, xgb_comprehensive_performance.png, svm_comprehensive_performance.png")
    print(f"- 所有算法汇总对比图: all_algorithms_summary_comparison.png")

if __name__ == "__main__":
    main()
