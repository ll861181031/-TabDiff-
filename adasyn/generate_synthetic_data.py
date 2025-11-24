"""
使用ADASYN生成合成数据，支持互斥列约束
"""

import pandas as pd
import numpy as np
import json
import argparse
from adasyn import Adasyn
from sklearn.preprocessing import LabelEncoder

# 设置随机种子以确保可重现性
np.random.seed(42)


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


def add_noise_to_synthetic_data(df, cat_col_names, num_col_names,
                                 cat_noise_prob=0.50, num_noise_std=0.30):
    """为合成数据添加受控噪声以增加多样性和真实性"""
    noisy_df = df.copy()

    # 为分类列添加噪声
    for col in cat_col_names:
        if col in noisy_df.columns:
            # 获取该列的唯一值
            unique_vals = noisy_df[col].unique()

            # 对每行随机决定是否添加噪声
            mask = np.random.random(len(noisy_df)) < cat_noise_prob
            n_changes = mask.sum()

            if n_changes > 0:
                # 从现有唯一值中随机选择新值
                new_vals = np.random.choice(unique_vals, size=n_changes)
                noisy_df.loc[mask, col] = new_vals

    # 为数值列添加噪声
    for col in num_col_names:
        if col in noisy_df.columns:
            # 计算列标准差
            col_std = noisy_df[col].std()

            if col_std > 0:
                # 添加高斯噪声
                noise = np.random.normal(0, col_std * num_noise_std, size=len(noisy_df))
                noisy_df[col] = noisy_df[col] + noise

                # 确保某些列为非负值
                if col in ['Posted Speed Limit (MPH)', 'SV Precrash Speed (MPH)',
                          'Model Year', 'Mileage']:
                    noisy_df[col] = np.maximum(0, noisy_df[col])

    return noisy_df


def generate_synthetic_samples(data, target_labels, dth=0.9, b=1.0, k=5, mutually_exclusive_groups=None):
    """使用ADASYN生成合成样本"""
    # 确保k不超过样本数量
    k = min(k, len(data) - 1)

    # 创建ADASYN实例
    adasyn = Adasyn(data, target_labels, dth=dth, b=b, K=k, mutually_exclusive_groups=mutually_exclusive_groups)

    # 生成合成样本
    adasyn.sampling()

    return np.array(adasyn.synthetic)


def _apply_mutually_exclusive_constraints_standalone(sample, mutually_exclusive_groups):
    """对样本应用互斥约束"""
    constrained_sample = sample.copy()

    for group in mutually_exclusive_groups:
        if len(group) > 0:
            # 获取该组的值
            group_values = constrained_sample[group]

            # 找到最大值的索引
            max_idx = np.argmax(group_values)

            for i, col_idx in enumerate(group):
                if i == max_idx:
                    constrained_sample[col_idx] = 1
                else:
                    constrained_sample[col_idx] = 0

    return constrained_sample


def process_by_class(df, target_col_name, feature_columns, target_samples=None, dth=0.9, b=1.0, k=5, mutually_exclusive_groups=None):
    """使用ADASYN处理类别不平衡，为每个类别生成样本以达到目标数量"""
    synthetic_samples = []
    synthetic_targets = []

    # 获取唯一类别及其计数
    classes = df[target_col_name].unique()
    class_counts = df[target_col_name].value_counts()

    print(f"处理 {len(classes)} 个类别...")
    print(f"类别分布:\n{class_counts}")

    # 确定每个类别的目标数量
    if target_samples is None:
        target_samples = class_counts.max()

    print(f"\n每个类别目标样本数: {target_samples}")

    # 处理需要更多样本的每个类别
    for class_label in classes:
        current_count = class_counts[class_label]
        samples_needed = target_samples - current_count

        print(f"\n类别 '{class_label}': {current_count} 个样本")

        if samples_needed <= 0:
            print(f"  无需生成样本 (已达到或超过目标)")
            continue

        print(f"  需要生成 {samples_needed} 个样本")

        # 获取当前类别样本
        class_df = df[df[target_col_name] == class_label]
        class_data = class_df[feature_columns].values

        # 调整k值
        k_adjusted = min(k, current_count - 1)
        if k_adjusted < 1:
            print(f"  样本太少无法使用ADASYN，使用简单过采样加噪声")
            generated_count = 0
            synthetic_data_list = []

            while generated_count < samples_needed:
                # 随机选择一个现有样本
                idx = np.random.randint(0, current_count)
                base_sample = class_data[idx].copy()

                # 添加随机噪声
                for col_idx in range(len(base_sample)):
                    # 为数值特征添加小随机噪声
                    if col_idx not in [idx for group in mutually_exclusive_groups for idx in group]:
                        noise = np.random.normal(0, 0.1)
                        base_sample[col_idx] = base_sample[col_idx] + noise

                # 应用互斥约束
                base_sample = _apply_mutually_exclusive_constraints_standalone(
                    base_sample, mutually_exclusive_groups
                )

                synthetic_data_list.append(base_sample)
                generated_count += 1

            synthetic_data = np.array(synthetic_data_list)
            print(f"  使用过采样生成了 {len(synthetic_data)} 个合成样本")

            synthetic_samples.append(synthetic_data)
            synthetic_targets.extend([class_label] * len(synthetic_data))
            continue

        # 获取其他类别样本用于ADASYN
        other_df = df[df[target_col_name] != class_label]
        other_data = other_df[feature_columns].values
        other_count = len(other_data)

        # 为ADASYN准备数据（二分类）
        X_combined = np.vstack([class_data, other_data])
        y_combined = np.array([1] * current_count + [-1] * other_count)

        # 计算生成所需样本所需的平衡级别
        ml = max(current_count, other_count)
        ms = min(current_count, other_count)

        if ml > ms:
            b_adjusted = samples_needed / (ml - ms)
            b_adjusted = min(2.0, b_adjusted)
        else:
            b_adjusted = b

        # 生成合成样本
        try:
            synthetic_data = generate_synthetic_samples(
                X_combined,
                y_combined,
                dth=dth,
                b=b_adjusted,
                k=k_adjusted,
                mutually_exclusive_groups=mutually_exclusive_groups
            )

            generated_count = len(synthetic_data)

            if generated_count > 0:
                # 如果生成过多，下采样
                if generated_count > samples_needed:
                    indices = np.random.choice(generated_count, samples_needed, replace=False)
                    synthetic_data = synthetic_data[indices]
                # 如果生成不足，过采样填补缺口
                elif generated_count < samples_needed:
                    gap = samples_needed - generated_count
                    print(f"  ADASYN生成了 {generated_count} 个样本，用过采样填补 {gap} 个缺口")

                    # 从现有类别样本中过采样
                    oversample_indices = np.random.choice(current_count, gap, replace=True)
                    oversampled_data = class_data[oversample_indices].copy()

                    # 为过采样数据添加噪声
                    for i in range(len(oversampled_data)):
                        noise = np.random.normal(0, 0.1, size=oversampled_data[i].shape)
                        oversampled_data[i] = oversampled_data[i] + noise
                        oversampled_data[i] = _apply_mutually_exclusive_constraints_standalone(
                            oversampled_data[i], mutually_exclusive_groups
                        )

                    # 合并ADASYN和过采样数据
                    synthetic_data = np.vstack([synthetic_data, oversampled_data])

                print(f"  生成了 {len(synthetic_data)} 个合成样本")

                # 存储合成样本及其目标
                synthetic_samples.append(synthetic_data)
                synthetic_targets.extend([class_label] * len(synthetic_data))
            else:
                print(f"  ADASYN生成了0个样本，改用过采样")
                # 如果ADASYN失败，回退到过采样
                oversample_indices = np.random.choice(current_count, samples_needed, replace=True)
                oversampled_data = class_data[oversample_indices].copy()

                # 添加噪声
                for i in range(len(oversampled_data)):
                    noise = np.random.normal(0, 0.1, size=oversampled_data[i].shape)
                    oversampled_data[i] = oversampled_data[i] + noise
                    oversampled_data[i] = _apply_mutually_exclusive_constraints_standalone(
                        oversampled_data[i], mutually_exclusive_groups
                    )

                synthetic_samples.append(oversampled_data)
                synthetic_targets.extend([class_label] * len(oversampled_data))
                print(f"  使用过采样生成了 {len(oversampled_data)} 个合成样本")

        except ValueError as e:
            print(f"  使用ADASYN生成样本时出错: {e}")
            print(f"  回退到过采样")
            # 回退到过采样
            oversample_indices = np.random.choice(current_count, samples_needed, replace=True)
            oversampled_data = class_data[oversample_indices].copy()

            # 添加噪声
            for i in range(len(oversampled_data)):
                noise = np.random.normal(0, 0.1, size=oversampled_data[i].shape)
                oversampled_data[i] = oversampled_data[i] + noise
                oversampled_data[i] = _apply_mutually_exclusive_constraints_standalone(
                    oversampled_data[i], mutually_exclusive_groups
                )

            synthetic_samples.append(oversampled_data)
            synthetic_targets.extend([class_label] * len(oversampled_data))
            print(f"  使用过采样生成了 {len(oversampled_data)} 个合成样本")

    # 合并所有合成样本
    if synthetic_samples:
        all_synthetic = np.vstack(synthetic_samples)

        # 创建数据框
        synthetic_df = pd.DataFrame(all_synthetic, columns=feature_columns)
        synthetic_df[target_col_name] = synthetic_targets

        return synthetic_df
    else:
        return pd.DataFrame()


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='使用ADASYN生成合成数据')
    parser.add_argument('--weather-exclusive', 
                       type=lambda x: x.lower() in ['true', '1', 'yes', 'on'],
                       default=True,
                       help='天气条件是否应该互斥 (默认: True)')
    parser.add_argument('--config', 
                       type=str, 
                       default='data/avoid.json',
                       help='配置文件路径 (默认: data/avoid.json)')
    parser.add_argument('--output-dir', 
                       type=str, 
                       default='data',
                       help='生成文件的输出目录 (默认: data)')
    parser.add_argument('--target-samples',
                       type=int,
                       default=None,
                       help='每个类别的目标样本数 (默认: 最大类别计数)')
    parser.add_argument('--dth', 
                       type=float, 
                       default=0.5,
                       help='最大容忍的不平衡比率 (默认: 0.5)')
    parser.add_argument('--b', 
                       type=float, 
                       default=1.0,
                       help='期望的平衡级别 (默认: 1.0)')
    parser.add_argument('--k', 
                       type=int, 
                       default=5,
                       help='最近邻数量 (默认: 5)')
    parser.add_argument('--cat-noise-prob', 
                       type=float, 
                       default=0.50,
                       help='分类噪声概率 (默认: 0.50)')
    parser.add_argument('--num-noise-std', 
                       type=float, 
                       default=0.30,
                       help='数值噪声标准差乘数 (默认: 0.30)')
    
    return parser.parse_args()


def main():
    """主函数：生成合成数据集"""
    
    # 解析命令行参数
    args = parse_arguments()
    
    print("="*60)
    print("ADASYN合成数据生成")
    print("="*60)
    print(f"天气互斥性: {args.weather_exclusive}")
    print(f"配置文件: {args.config}")
    print(f"输出目录: {args.output_dir}")
    print(f"目标样本: {args.target_samples if args.target_samples else '与原始相同'}")
    print(f"不平衡阈值 (dth): {args.dth}")
    print(f"平衡级别 (b): {args.b}")
    print(f"最近邻 (k): {args.k}")
    print(f"分类噪声概率: {args.cat_noise_prob}")
    print(f"数值噪声标准差乘数: {args.num_noise_std}")
    print("="*60)

    # 加载配置
    print("加载配置...")
    config = load_config(args.config)

    # 加载训练数据
    train_path = f'{args.output_dir}/train.csv'
    print(f"从 {train_path} 加载训练数据...")
    train_df = load_data(train_path)
    print(f"加载了 {len(train_df)} 个训练样本")
    print(f"列数: {len(train_df.columns)}")

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

    # 定义天气列
    weather_col_names = [
        'Weather - Clear', 'Weather - Rain', 'Weather - Snow',
        'Weather - Cloudy', 'Weather - Fog/Smoke', 'Weather - Severe Wind'
    ]

    # 获取天气列索引
    feature_columns = [col for col in encoded_df.columns if col != target_col_name]
    weather_col_indices = [feature_columns.index(col) for col in weather_col_names if col in feature_columns]

    print(f"\n天气列: {weather_col_names}")
    print(f"天气列索引: {weather_col_indices}")

    # 根据命令行参数定义互斥组
    if args.weather_exclusive:
        mutually_exclusive_groups = [weather_col_indices]
        print("天气条件将互斥")
    else:
        mutually_exclusive_groups = []
        print("天气条件将不互斥")

    # 生成合成数据
    print("\n" + "="*50)
    print("使用ADASYN生成合成数据...")
    print("="*50)

    # 设置每个类别的目标样本数
    target_samples_per_class = args.target_samples
    if target_samples_per_class:
        print(f"每个类别目标样本数: {target_samples_per_class}")
    else:
        print(f"每个类别目标样本数: 自动 (最大类别计数)")

    synthetic_df = process_by_class(
        encoded_df,
        target_col_name=target_col_name,
        feature_columns=feature_columns,
        target_samples=target_samples_per_class,
        dth=args.dth,
        b=args.b,
        k=args.k,
        mutually_exclusive_groups=mutually_exclusive_groups
    )

    if len(synthetic_df) > 0:
        # 为合成数据添加噪声以增加多样性
        print("\n为合成数据添加受控噪声...")
        print(f"  - 分类噪声概率: {args.cat_noise_prob*100:.0f}%")
        print(f"  - 数值噪声标准差乘数: {args.num_noise_std}")

        # 识别天气列以排除在一般分类噪声之外
        weather_col_names_in_features = [col for col in weather_col_names if col in feature_columns]

        # 获取非天气分类列
        cat_cols_for_noise = [col for col in cat_col_names if col in feature_columns
                              and col not in weather_col_names_in_features]

        # 添加噪声（但如果启用则保持天气互斥性）
        synthetic_df_noisy = add_noise_to_synthetic_data(
            synthetic_df,
            cat_col_names=cat_cols_for_noise,  # 排除天气列
            num_col_names=num_col_names,
            cat_noise_prob=args.cat_noise_prob,
            num_noise_std=args.num_noise_std
        )

        # 在噪声后重新应用天气互斥性（仅在启用时）
        if args.weather_exclusive and len(weather_col_indices) > 0:
            print("\n重新应用天气互斥性约束...")
            for idx in range(len(synthetic_df_noisy)):
                weather_values = synthetic_df_noisy.iloc[idx, weather_col_indices].values
                max_idx = np.argmax(weather_values)
                for i, col_idx in enumerate(weather_col_indices):
                    synthetic_df_noisy.iloc[idx, feature_columns.index(feature_columns[col_idx])] = 1 if i == max_idx else 0

        # 将分类列解码回原始值
        print("\n解码分类列...")
        synthetic_df_decoded = decode_categorical_columns(synthetic_df_noisy, encoders)

        # 合并原始和合成数据
        combined_df = pd.concat([train_df, synthetic_df_decoded], ignore_index=True)

        # 保存合成数据
        synthetic_output_path = f'{args.output_dir}/synthetic_train.csv'
        synthetic_df_decoded.to_csv(synthetic_output_path, index=False)
        print(f"\n合成数据保存到: {synthetic_output_path}")
        print(f"合成样本数: {len(synthetic_df_decoded)}")

        # 保存合并数据
        combined_output_path = f'{args.output_dir}/combined_train.csv'
        combined_df.to_csv(combined_output_path, index=False)
        print(f"\n合并数据保存到: {combined_output_path}")
        print(f"总样本数 (原始 + 合成): {len(combined_df)}")

        # 打印类别分布
        print("\n" + "="*50)
        print("合成数据中的类别分布:")
        print("="*50)
        print(synthetic_df_decoded[target_col_name].value_counts())

        print("\n" + "="*50)
        print("合并数据中的类别分布:")
        print("="*50)
        print(combined_df[target_col_name].value_counts())

        # 验证天气列约束（仅在启用互斥性时）
        if args.weather_exclusive:
            print("\n" + "="*50)
            print("验证天气列约束...")
            print("="*50)
            # 将Y转换为1进行验证
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
            print("\n天气互斥性已禁用 - 无需约束验证")

    else:
        print("\n未生成合成数据。")


if __name__ == '__main__':
    main()