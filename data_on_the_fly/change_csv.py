import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import ast  # 用于字面值评估（literal_eval）的模块


def scale_data(input_file, output_file, column_name, target_min, target_max, decimal_places=8):
    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 将包含列表的字符串转换为实际的列表
    df[column_name] = df[column_name].apply(ast.literal_eval)

    # 将列表展平成一个单一的列表
    data = [item for sublist in df[column_name] for item in sublist]

    # 将数据重新形状为列向量
    data = pd.DataFrame(data, columns=[column_name])

    # 使用MinMaxScaler进行缩放
    scaler = MinMaxScaler(feature_range=(target_min, target_max))
    scaled_data = scaler.fit_transform(data)

    # 将缩放后的数据四舍五入到指定的小数位数
    scaled_data_str = scaled_data.round(decimal_places)

    # 将缩放后的数据添加到DataFrame中
    df[f'{column_name}_scaled'] = scaled_data_str

    # 删除原始的'MDS'列
    df.drop(column_name, axis=1, inplace=True)

    # 保存到新的CSV文件
    df.to_csv(output_file, index=False)


# 示例用法
input_file = 'mds_1_orig.csv'
output_file = 'mds_1_dim.csv'
column_name = 'MDS'  # 假设数值列为 'MDS'
target_min = 1
target_max = 103
decimal_places = 8

# 调用函数
scale_data(input_file, output_file, column_name, target_min, target_max, decimal_places)
