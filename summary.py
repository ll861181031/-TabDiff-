'''
汇总各个分类算法在不同数据增强方法下的指标包括大类和小类
'''
import pandas as pd
import numpy as np
from pathlib import Path

def load_data():
    df_all = pd.read_csv("results/plots/model_all.csv")
    df_class = pd.read_csv("results/plots/model.csv")
    
    df_all = df_all.dropna(subset=['Model'])
    df_all['Original'] = df_all['Original'].ffill()
    if pd.isna(df_all.iloc[0]['Original']):
        df_all.iloc[0, df_all.columns.get_loc('Original')] = 'Original'
        df_all['Original'] = df_all['Original'].ffill()
    
    df_class = df_class.dropna(subset=['Model'])
    df_class['Original'] = df_class['Original'].ffill()
    if pd.isna(df_class.iloc[0]['Original']):
        df_class.iloc[0, df_class.columns.get_loc('Original')] = 'Original'
        df_class['Original'] = df_class['Original'].ffill()
    
    return df_all, df_class

def calculate_g_mean_for_algorithm(df_class, algorithm, dataset):
    alg_data = df_class[(df_class['Model'] == algorithm) & (df_class['Original'] == dataset)]
    if len(alg_data) == 0:
        return 0.0
    
    g_means = []
    for _, row in alg_data.iterrows():
        g_means.append(row['G_mean'])
    
    return np.mean(g_means) if g_means else 0.0

def get_class_recall(df_class, algorithm, dataset, class_num):
    alg_data = df_class[(df_class['Model'] == algorithm) & 
                       (df_class['Original'] == dataset) & 
                       (df_class['Class'] == class_num)]
    return alg_data['Recall'].iloc[0] if len(alg_data) > 0 else 0.0

def create_algorithm_summary_csv(df_all, df_class, algorithm, output_dir):
    datasets = ['Original', 'Smote', 'CtGan', 'AdaSyn', 'TabDiff']
    metrics = [
        'Total Accuracy',
        'Accuracy_class 0', 
        'Accuracy_class 1',
        'Accuracy_class 2',
        'Accuracy_class 4',
        'F1-Score',
        'G-mean',
        'AUC'
    ]
    
    summary_data = []
    
    for metric in metrics:
        row_data = [metric]
        
        for dataset in datasets:
            if metric == 'Total Accuracy':
                alg_data = df_all[(df_all['Model'] == algorithm) & (df_all['Original'] == dataset)]
                value = alg_data['Accuracy'].iloc[0] if len(alg_data) > 0 else 0.0
                    
            elif metric == 'Accuracy_class 0':
                value = get_class_recall(df_class, algorithm, dataset, 0)
                
            elif metric == 'Accuracy_class 1':
                value = get_class_recall(df_class, algorithm, dataset, 1)
                
            elif metric == 'Accuracy_class 2':
                value = get_class_recall(df_class, algorithm, dataset, 2)
                
            elif metric == 'Accuracy_class 4':
                value = get_class_recall(df_class, algorithm, dataset, 4)
                
            elif metric == 'F1-Score':
                alg_data = df_all[(df_all['Model'] == algorithm) & (df_all['Original'] == dataset)]
                value = alg_data['F1-Weighted'].iloc[0] if len(alg_data) > 0 else 0.0
                    
            elif metric == 'G-mean':
                value = calculate_g_mean_for_algorithm(df_class, algorithm, dataset)
                
            elif metric == 'AUC':
                alg_data = df_all[(df_all['Model'] == algorithm) & (df_all['Original'] == dataset)]
                value = alg_data['AUC'].iloc[0] if len(alg_data) > 0 else 0.0
            
            row_data.append(round(value, 4))
        
        summary_data.append(row_data)
    
    columns = ['Metric'] + datasets
    df_summary = pd.DataFrame(summary_data, columns=columns)
    
    csv_path = output_dir / f"{algorithm.lower()}_summary.csv"
    df_summary.to_csv(csv_path, index=False, encoding='utf-8-sig')
    
    print(f"{algorithm}汇总CSV文件已保存到: {csv_path}")
    print(f"\n{algorithm}汇总数据预览:")
    print(df_summary.to_string(index=False))
    
    return df_summary

def main():
    output_dir = Path("results/plots")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("正在加载数据...")
    df_all, df_class = load_data()
    
    algorithms = df_all['Model'].unique()
    print(f"\n将为以下算法创建汇总CSV文件: {list(algorithms)}")
    
    for algorithm in algorithms:
        print(f"\n正在创建{algorithm}算法的汇总CSV文件...")
        create_algorithm_summary_csv(df_all, df_class, algorithm, output_dir)
    
    print(f"\n所有算法汇总CSV文件已成功保存到 results/plots/ 文件夹中！")
    print(f"\n生成的文件包括:")
    for algorithm in algorithms:
        print(f"- {algorithm.lower()}_summary.csv")

if __name__ == "__main__":
    main()
