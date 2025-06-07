import torch

# 部分方向翻转
def partial_flip_direction(data, flip_factor):
    flipped_data = data.clone()
    num_samples = data.size(0)

    for i in range(num_samples):
        # 获取非零特征的索引
        non_zero_indices = torch.nonzero(data[i]).squeeze(dim=1)
        num_non_zero = non_zero_indices.size(0)

        if num_non_zero > 0:
            # 计算该样本要翻转的特征点数，不超过非零特征的数量
            num_flip_points = min(int(num_non_zero * flip_factor), num_non_zero)

            # 如果有非零特征，则从中随机选择要翻转的特征
            flip_indices = non_zero_indices[torch.randperm(num_non_zero)[:num_flip_points]]
            flipped_data[i, flip_indices] = -flipped_data[i, flip_indices]

    return flipped_data


# 数据增强
def augmented_data(data, flip_factor=0.05):
    augmented_data = data.clone()

    # 随机进行方向翻转
    augmented_data = partial_flip_direction(augmented_data, flip_factor)
    return augmented_data


def divide_array_by_sizes(arr, a, b, c):
    # 确保a + b + c等于数组的长度
    if len(arr) != a + b + c:
        raise ValueError("The sum of a, b, and c must equal the length of the array.")

    # 分割数组
    part1 = arr[:a]
    part2 = arr[a:a + b]
    part3 = arr[a + b:a+b+c]

    return part1, part2, part3
