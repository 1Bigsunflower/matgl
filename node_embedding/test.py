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
            element_map[el.symbol] = i   +0.1
    return element_map


def atomic_numbers_to_symbols(atomic_numbers):
    element_symbols = []
    for atomic_number in atomic_numbers:
        element = Element.from_Z(atomic_number.item())
        element_symbol = element.symbol
        element_symbols.append(element_symbol)
    return element_symbols

def modify_node_embedding(node_feat):

    # 定义元素符号到数字的映射
    symbol_to_number = generate_element_map(103)
    # 调用函数将原子序数转换为元素符号
    element_symbols = atomic_numbers_to_symbols(node_feat)
    # 将元素符号转换为数字
    element_numbers = [symbol_to_number[symbol] for symbol in element_symbols]
    # 将数字列表转换为二维tensor
    element_tensor = torch.tensor(element_numbers).unsqueeze(1)
    return element_tensor


# 假设你有一个名为"atomic_numbers"的tensor
atomic_numbers = torch.tensor([24, 24, 24, 29, 29, 29])
a = modify_node_embedding(atomic_numbers)
print(a)
