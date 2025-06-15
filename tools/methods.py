import torch

# Perform a direction flip.
def partial_flip_direction(data, flip_factor):
    flipped_data = data.clone()
    num_samples = data.size(0)

    for i in range(num_samples):
        # Get the index of the non-zero feature.
        non_zero_indices = torch.nonzero(data[i]).squeeze(dim=1)
        num_non_zero = non_zero_indices.size(0)

        if num_non_zero > 0:
            # Calculate the number of feature points to flip for
            # this sample, not exceeding the number of non-zero features.
            num_flip_points = min(int(num_non_zero * flip_factor), num_non_zero)

            # If there are non-zero features, randomly select from them the features to be flipped.
            flip_indices = non_zero_indices[torch.randperm(num_non_zero)[:num_flip_points]]
            flipped_data[i, flip_indices] = -flipped_data[i, flip_indices]

    return flipped_data


# Data enhancement.
def augmented_data(data, flip_factor=0.05):
    augmented_data = data.clone()

    # Random direction flips.
    augmented_data = partial_flip_direction(augmented_data, flip_factor)
    return augmented_data


def divide_array_by_sizes(arr, a, b, c):
    if len(arr) != a + b + c:
        raise ValueError("The sum of a, b, and c must equal the length of the array.")
    part1 = arr[:a]
    part2 = arr[a:a + b]
    part3 = arr[a + b:a+b+c]

    return part1, part2, part3
