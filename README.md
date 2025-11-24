
### 目录结构
```
生成对抗表格/
├─ data/                      
│  ├─ avoid.json              # 列索引与互斥组配置等
│  └─ split_data/split_data/  # train_original.csv 等原始划分
├─ results/
│  └─ plots/                  # 各模型生成的图表与CSV其中model_all是各个模型总的评价指标,model是各个模型小类的指标汇总表格
├─ smote/
│  ├─ generate_synthetic_data.py          # SMOTE 合成
│  └─ marginal_correlation_evaluation.py  # 边际/相关性评估
├─ CTGAN/
│  ├─ generate_synthetic_data.py          # CTGAN 合成
│  └─ marginal_correlation_evaluation.py           # 边际/相关性评估
├─ adasyn/
│  ├─ generate_synthetic_data.py          # ADASYN 合成
│  └─ marginal_correlation_evaluation.py  # 边际/相关性评估
├─ TabDiff/
│  ├─ process_dataset.py                  # 数据预处理
│  ├─ utils_train.py                      # 训练工具与封装
│  └─ main.py                             # 训练主入口
├─ tabdiff_evaluation.py      # 使用合成数据训练、原始测试集评估 + SHAP+roc曲线
├─ orignial_evaluation.py     # 原始数据训练与评估 + SHAP+roc曲线
├─ synthetic_data_spearman.py # 合成 vs 原始 Spearman 相关性差值热力图
├─ synthetic_data_comparison.py# 合成方法边际/相关性得分对比
├─ models_charts.py           # 各模型跨方法性能对比图 + 汇总图
├─ summary.py                 # 汇总生成各算法的 summary CSV
└─ bayesian_optimization.py   # 超参搜索脚本
```


### 运行步骤
准备 `data/avoid.json` 与原始 CSV
1) 生成 adasyn 合成数据（支持互斥列约束）
```bash
不启用weather列互斥
python generate_synthetic_data.py --dth 0.1 --b 1.0 --k 5 --weather-exclusive false
启用weather
python generate_synthetic_data.py --dth 0.1 --b 1.0 --k 5
```

```bash
边际得分以及相关性得分
python marginal_correlation_evaluation.py
```


2) 生成 CTGAN 合成数据（支持互斥列约束）
```bash
默认互斥是关闭的
python train_avoid_dataset.py 
启用互斥
python train_avoid_dataset.py --weather-exclusivity true 
```

```bash
边际得分以及相关性得分
python marginal_correlation_evaluation.py
```

3) 生成 smote 合成数据（支持互斥列约束）
```bash
generate_synthetic_data.py代码修改 enforce_weather_exclusivity 变量：
#设置为 True：weather 列互斥（每个样本只能有一个 weather 条件）
enforce_weather_exclusivity = True
# 设置为 False：weather 列不互斥（允许多个 weather 条件同时存在）
enforce_weather_exclusivity = False
python generate_synthetic_data.py 
```

```bash
边际得分以及相关性得分
python marginal_correlation_evaluation.py
```

4) 生成 TabDiff 合成数据（支持互斥列约束）
```bash
数据预处理
python process_dataset.py
训练模型
python main.py --dataname avoid --mode train --gpu 0 --no_wandb --weather_exclusive
生成合成数据并计算相关性得分和边际得分
python main.py \
      --dataname avoid \
      --mode test \
      --gpu 0 \
      --balance_target supplement \
      --weather_exclusive \
      --num_samples_to_generate 1785 \
      -no_wandb
```

5) 可视化
1. 运行 `smote/generate_synthetic_data.py` 生成 SMOTE 合成数据
2. 运行 `synthetic_data_spearman.py` 生成相关性差值热力图
3. 运行 `tabdiff_evaluation.py` 与/或 `orignial_evaluation.py` 对各个模型做评估并生成shap可视化，以及roc
4. 运行 `summary.py` 生成各算法汇总 CSV
5. 运行 `models_charts.py` 生成综合对比图
6. 运行 `synthetic_data_comparison.py` 相关性得分和边际分布得分对比图

### results可视化（部分）
- `results/plots/lr_comprehensive_performance.png` 等：按算法的综合性能对比图（Original/Smote/CtGan/AdaSyn/TabDiff）
- `results/plots/all_algorithms_summary_comparison.png`：总体汇总对比图
- `results/plots/spearman_correlation_difference_*.png`：合成 vs 原始 Spearman 相关性差值热力图
- `results/plots/*_summary.csv`：各算法指标汇总（由 `summary.py` 生成）
- `results/plots/synthetic_data_comparison.png`：不同合成方法边际/相关性得分对比示例
- `results/roc/adasyn`：不同合成方法的roc曲线
- `results/shap/adasyn`：不同合成方法的shap可视化
- `results/roc/adasyn`：不同合成方法的roc曲线
- `results/smote/evaluation_results.csv`：smote合成数据总的评价指标
- `results/smote/per_class_metrics.csv`：smote合成数据各个小类的评价指标
- `results/tabdiff/evaluation_results.csv`：tabdiff合成数据总的评价指标
- `results/tabdiff/per_class_metrics.csv`：tabdiff合成数据各个小类的评价指标
- `results/adasyn/evaluation_results.csv`：adasyn合成数据总的评价指标
- `results/adasyn/per_class_metrics.csv`：adasyn合成数据各个小类的评价指标
- `results/ctgan/evaluation_results.csv`：ctgan合成数据总的评价指标
- `results/ctgan/per_class_metrics.csv`：ctgan合成数据各个小类的评价指标





