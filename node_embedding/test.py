from pymatgen.core import Element
import torch


def generate_element_map(max_atomic_number):
    """
    生成一个将化学元素名称映射到原子序数的字典

    参数:
    max_atomic_number (int): 要生成的原子序数的最大值

    返回:
    dict: 包含化学元素名称映射到原子序数的字典
    """
    element_map = {}
    for i in range(1, max_atomic_number+1):
        el = Element.from_Z(i)
        if el.symbol:
            element_map[el.symbol] = i+0.1
    return element_map


# def atomic_numbers_to_symbols(atomic_numbers):
#     element_symbols = []
#     for atomic_number in atomic_numbers:
#         element = Element.from_Z(atomic_number.item())
#         element_symbol = element.symbol
#         element_symbols.append(element_symbol)
#     return element_symbols




# generate_element_map(103)
import csv

# def generate_element_map_from_csv(csv_filepath):
#     """
#     从CSV文件生成一个将化学元素名称映射到映射值的字典
#
#     参数:
#     csv_filepath (str): CSV文件的路径
#
#     返回:
#     dict: 包含化学元素名称映射到映射值的字典
#     """
#     element_map = {}
#     with open(csv_filepath, mode='r', encoding='utf-8') as csvfile:
#         reader = csv.reader(csvfile)
#         next(reader)  # 跳过标题行
#         for row in reader:
#             if len(row) == 2:  # 确保每行有两个元素
#                 element, value = row
#                 # 假设value是字符串形式的列表，我们需要将其转换为实际的列表对象
#                 # 这里使用eval来将字符串转换成Python对象，但在实际应用中需要确保安全性
#                 try:
#                     value = eval(value)
#                 except (SyntaxError, NameError):
#                     continue  # 如果转换失败，跳过这一行
#                 if isinstance(value, list) and len(value) == 1:
#                     element_map[element] = round(value[0], 3)  # 只取列表中的单一元素
#     return element_map





# def atomic_numbers_to_symbols(atomic_numbers):  # 输入node_feat，转化为化学元素
#     element_symbols = []
#     for atomic_number in atomic_numbers:
#         element = Element.from_Z(atomic_number.item() + 1)
#         element_symbol = element.symbol
#         element_symbols.append(element_symbol)
#     return element_symbols
#
# # 假设你有一个名为"atomic_numbers"的tensor
# atomic_numbers = torch.tensor([24, 24, 24, 29, 29, 29])
# a = atomic_numbers_to_symbols(atomic_numbers)
# print(a)

def generate_element_map_from_csv(csv_filepath):
    """
    从CSV文件生成一个将化学元素名称映射到出现顺序的映射字典

    参数:
    csv_filepath (str): CSV文件的路径

    返回:
    dict: 包含化学元素名称映射到出现顺序的映射字典
    """
    element_map = {}
    with open(csv_filepath, mode='r', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳过标题行
        for index, row in enumerate(reader, start=1):  # 从1开始计数
            if len(row) == 2:  # 确保每行有两个元素
                element, _ = row  # 只需要化学元素名称，不需要对应的值
                element_map[element] = index - 1  # 减去 1，不计算标题行
    return element_map

csv_path = '../node_embedding_new/sorted_data-mds-11.17.csv'  # 替换为您的CSV文件路径
element_map = generate_element_map_from_csv(csv_path)
print(element_map)