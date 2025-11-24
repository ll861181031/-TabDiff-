"""
使用SMOTE生成合成数据，支持互斥列约束
"""

import pandas as pd
import numpy as np
import json
from smote import Smote
from sklearn.preprocessing import LabelEncoder


def load_config(config_path='data/avoid.json'):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return json.load(f)


def load_data(csv_path):
    """加载CSV数据"""
    return pd.read_csv(csv_path)


def encode_categorical_columns(df, cat_col_names):
    """使用LabelEncoder编码分类列"""
    encoded_df = df.copy()
    encoders = {}

    for col in cat_col_names:
        if col in encoded_df.columns:
            # 处理缺失值
            encoded_df[col] = encoded_df[col].fillna('Unknown')
            # 编码
            le = LabelEncoder()
            encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
            encoders[col] = le

    return encoded_df, encoders


def decode_categorical_columns(df, encoders):
    """将分类列解码回原始值"""
    decoded_df = df.copy()

    for col, le in encoders.items():
        if col in decoded_df.columns:
            # 四舍五入到最近的整数
            decoded_df[col] = np.round(decoded_df[col]).astype(int)
            # 限制值在有效范围内
            decoded_df[col] = np.clip(decoded_df[col], 0, len(le.classes_) - 1)
            # 解码
            decoded_df[col] = le.inverse_transform(decoded_df[col])

    return decoded_df


def generate_synthetic_samples(data, N=100, k=5, mutually_exclusive_groups=None,
                             enforce_exclusivity=True, categorical_indices=None, noise_scale=0.015):
    """使用SMOTE生成合成样本"""
    # 确保k不超过样本数量
    k = min(k, len(data) - 1)

    # 创建SMOTE实例
    smote = Smote(data, N=N, k=k, mutually_exclusive_groups=mutually_exclusive_groups,
                  enforce_exclusivity=enforce_exclusivity,
                  categorical_indices=categorical_indices,
                  noise_scale=noise_scale)

    # 生成合成样本
    smote.over_sampling()

    return np.array(smote.synthetic)


def process_by_class(df, target_col_name, feature_columns, N=100, k=5,
                    mutually_exclusive_groups=None, enforce_exclusivity=True,
                    categorical_indices=None, noise_scale=0.015):
    """对少数类应用SMOTE，平衡所有类到最大类大小"""
    synthetic_samples = []
    synthetic_targets = []

    # 获取唯一类和计数
    classes = df[target_col_name].unique()
    class_counts = df[target_col_name].value_counts()
    max_class_count = class_counts.max()

    print(f"处理 {len(classes)} 个类别...")
    print(f"最大类大小: {max_class_count}")
    print(f"目标: 平衡所有类到 {max_class_count} 个样本 (原始 + 合成)")

    for class_label in classes:
        # 获取该类的样本
        class_df = df[df[target_col_name] == class_label]
        class_data = class_df[feature_columns].values
        current_count = len(class_data)

        print(f"\n类别 '{class_label}': {current_count} 个原始样本")

        # 如果该类已经有最大样本数，跳过
        if current_count >= max_class_count:
            print(f"  已达到最大大小，无需生成合成样本")
            continue

        # 如果样本太少无法应用SMOTE，跳过
        if current_count < k + 1:
            print(f"  跳过 (少于 {k+1} 个样本，无法应用SMOTE)")
            continue

        # 计算需要生成多少合成样本
        samples_needed = max_class_count - current_count
        print(f"  需要生成 {samples_needed} 个合成样本以达到 {max_class_count} 个总数")

        # 计算需要的N百分比
        N_needed = int((samples_needed / current_count) * 100)
        print(f"  使用 N={N_needed}% (应生成约 {int(N_needed * current_count / 100)} 个样本)")

        # 生成合成样本
        synthetic_data = generate_synthetic_samples(
            class_data,
            N=N_needed,
            k=k,
            mutually_exclusive_groups=mutually_exclusive_groups,
            enforce_exclusivity=enforce_exclusivity,
            categorical_indices=categorical_indices,
            noise_scale=noise_scale
        )

        actual_generated = len(synthetic_data)
        print(f"  生成了 {actual_generated} 个合成样本")

        # 如果生成不够，生成更多样本
        if actual_generated < samples_needed:
            additional_needed = samples_needed - actual_generated
            print(f"  生成 {additional_needed} 个额外样本以达到目标")

            N_additional = int((additional_needed / current_count) * 100) + 10
            additional_data = generate_synthetic_samples(
                class_data,
                N=N_additional,
                k=k,
                mutually_exclusive_groups=mutually_exclusive_groups,
                enforce_exclusivity=enforce_exclusivity,
                categorical_indices=categorical_indices,
                noise_scale=noise_scale
            )

            additional_data = additional_data[:additional_needed]
            synthetic_data = np.vstack([synthetic_data, additional_data])
            print(f"  总生成: {len(synthetic_data)} 个合成样本")
        elif actual_generated > samples_needed:
            # 如果生成太多，修剪到确切数量
            print(f"  修剪到恰好 {samples_needed} 个样本")
            synthetic_data = synthetic_data[:samples_needed]

        # 存储合成样本和目标
        synthetic_samples.append(synthetic_data)
        synthetic_targets.extend([class_label] * len(synthetic_data))

        print(f"  最终: {current_count} 个原始 + {len(synthetic_data)} 个合成 = {current_count + len(synthetic_data)} 个总数")

    # 合并所有合成样本
    if synthetic_samples:
        all_synthetic = np.vstack(synthetic_samples)
        synthetic_df = pd.DataFrame(all_synthetic, columns=feature_columns)
        synthetic_df[target_col_name] = synthetic_targets
        return synthetic_df
    else:
        return pd.DataFrame()


def main():
    """主函数：生成合成数据集"""
    print("加载配置...")
    config = load_config('data/avoid.json')

    print("加载训练数据...")
    train_df = load_data('data/train.csv')
    print(f"加载了 {len(train_df)} 个训练样本，{len(train_df.columns)} 列")

    # 获取目标列名
    target_col_name = config['column_names'][config['target_col_idx'][0]]
    print(f"\n目标列: {target_col_name}")

    # 获取分类列名
    cat_col_names = [config['column_names'][idx] for idx in config['cat_col_idx']]
    print(f"\n分类列: {len(cat_col_names)} 列")

    # 编码分类列
    print("\n编码分类列...")
    encoded_df, encoders = encode_categorical_columns(train_df, cat_col_names)

    # 处理数值列 - 用均值填充NaN
    num_col_names = [config['column_names'][idx] for idx in config['num_col_idx']]
    for col in num_col_names:
        if col in encoded_df.columns:
            encoded_df[col] = encoded_df[col].fillna(encoded_df[col].mean())

    # 定义天气列（互斥组）
    weather_col_names = [
        'Weather - Clear', 'Weather - Rain', 'Weather - Snow',
        'Weather - Cloudy', 'Weather - Fog/Smoke', 'Weather - Severe Wind'
    ]

    # 获取特征列和天气列索引
    feature_columns = [col for col in encoded_df.columns if col != target_col_name]
    weather_col_indices = [feature_columns.index(col) for col in weather_col_names if col in feature_columns]

    print(f"\n天气列 (互斥): {weather_col_names}")
    print(f"天气列索引: {weather_col_indices}")

    # 获取分类特征索引
    categorical_feature_indices = []
    for col_name in cat_col_names:
        if col_name in feature_columns:
            categorical_feature_indices.append(feature_columns.index(col_name))

    print(f"\n分类特征索引: {len(categorical_feature_indices)} 列")

    # 定义互斥组
    mutually_exclusive_groups = [weather_col_indices]
    enforce_weather_exclusivity = True

    # SMOTE参数优化
    smote_N = 100          # 生成比例
    smote_k = 5            # 最近邻数量
    noise_scale = 0.025    # 噪声尺度

    # 生成合成数据
    print("\n" + "="*50)
    print("生成合成数据以平衡类别...")
    print(f"策略: 仅为少数类生成合成样本")
    print(f"目标: 每个类达到 {encoded_df[target_col_name].value_counts().max()} 个样本 (原始 + 合成)")
    print(f"天气互斥性强制: {enforce_weather_exclusivity}")
    print(f"SMOTE参数: N={smote_N}%, k={smote_k}, noise={noise_scale}")
    print("="*50)

    synthetic_df = process_by_class(
        encoded_df,
        target_col_name=target_col_name,
        feature_columns=feature_columns,
        N=smote_N,
        k=smote_k,
        mutually_exclusive_groups=mutually_exclusive_groups,
        enforce_exclusivity=enforce_weather_exclusivity,
        categorical_indices=categorical_feature_indices,
        noise_scale=noise_scale
    )

    if len(synthetic_df) > 0:
        # 解码分类列回原始值
        print("\n解码分类列...")
        synthetic_df_decoded = decode_categorical_columns(synthetic_df, encoders)

        # 合并原始和合成数据
        combined_df = pd.concat([train_df, synthetic_df_decoded], ignore_index=True)

        # 保存合成数据
        synthetic_output_path = 'data/synthetic_train.csv'
        synthetic_df_decoded.to_csv(synthetic_output_path, index=False)
        print(f"\n合成数据保存到: {synthetic_output_path}")
        print(f"生成的合成样本数: {len(synthetic_df_decoded)}")

        # 保存合并数据
        combined_output_path = 'data/combined_train.csv'
        combined_df.to_csv(combined_output_path, index=False)
        print(f"\n平衡数据集 (原始 + 合成) 保存到: {combined_output_path}")
        print(f"总样本数: {len(combined_df)} = {len(train_df)} 原始 + {len(synthetic_df_decoded)} 合成")

        # 打印类别分布
        print("\n" + "="*50)
        print("类别分布摘要:")
        print("="*50)
        print("\n原始训练数据:")
        print(train_df[target_col_name].value_counts().sort_index())

        print("\n生成的合成样本 (仅少数类):")
        print(synthetic_df_decoded[target_col_name].value_counts().sort_index())

        print("\n平衡的合并数据集 (原始 + 合成):")
        print(combined_df[target_col_name].value_counts().sort_index())

        # 验证天气列约束
        print("\n" + "="*50)
        print("验证天气列约束...")
        print("="*50)
        weather_data = synthetic_df_decoded[weather_col_names].copy()
        for col in weather_col_names:
            weather_data[col] = weather_data[col].apply(lambda x: 1 if str(x).strip().upper() == 'Y' else 0)
        weather_sum = weather_data.sum(axis=1)
        print(f"所有样本都恰好有一个天气条件: {(weather_sum == 1).all()}")
        print(f"天气和统计: min={weather_sum.min()}, max={weather_sum.max()}, mean={weather_sum.mean():.2f}")

        # 显示一些示例
        print("\n前5个合成样本 (天气列):")
        print(synthetic_df_decoded[weather_col_names].head())

    else:
        print("\n未生成合成数据。")


if __name__ == '__main__':
    main()