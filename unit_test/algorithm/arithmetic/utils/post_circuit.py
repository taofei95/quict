from numpy import binary_repr
from typing import List


def decode_counts(count_list: List[int], partition: List[int], count_thresh: int = 0) -> List[List[str]]:
    """
        Given a partition, decode and partition the sampling result from a simulator into a list of
        binary strings.

        Args:
            count_list (List[int]): list of sampling result from a simulator.

            partition (List[int]): a list of integer representing the lengths of individual
            binary string. The sum requires to be equal to the length of the count_list.

            count_thresh (int): only the index with count larger than count_thresh will be
            processed. Default to be 0.

        Return:
            A list of partitioned binary string.
    """
    bin_size = len(binary_repr(len(count_list))) - 1
    sum_of_parts = sum(partition)

    if sum_of_parts != bin_size:
        if sum_of_parts > bin_size:
            raise ValueError(f"toal decode register size {sum_of_parts} larger than bits measured: {bin_size}")
        else:
            raise ValueError(f"toal decode register size {sum_of_parts} smaller than bits measured: {bin_size}")

    res = []

    for idx, count in enumerate(count_list):
        if count > count_thresh:
            total_str = binary_repr(idx, bin_size)

            single_decoded_part = []

            start_idx = 0
            for part_size in partition:
                single_decoded_part.append(total_str[start_idx:start_idx + part_size])
                start_idx += part_size

            res.append(single_decoded_part)

    return res


def decode_counts_int(count_list: List[int], partition: List[int], count_thresh: int = 0) -> List[List[int]]:
    """
        Given a partition, decode and partition the sampling result from a simulator into a list of
        binary strings then converting into decimal number.

        Args:
            count_list (List[int]): list of sampling result from a simulator.

            partition (List[int]): a list of integer representing the lengths of individual
            binary string. The sum requires to be equal to the length of the count_list.

            count_thresh (int): only the index with count larger than count_thresh will be
            processed. Default to be 0.

        Return:
            A list of partitioned binary strings converted into decimal integers.
    """
    bin_size = len(binary_repr(len(count_list))) - 1
    sum_of_parts = sum(partition)

    if sum_of_parts != bin_size:
        if sum_of_parts > bin_size:
            raise ValueError(f"toal decode register size {sum_of_parts} larger than bits measured: {bin_size}")
        else:
            raise ValueError(f"toal decode register size {sum_of_parts} smaller than bits measured: {bin_size}")

    res = []

    for idx, count in enumerate(count_list):
        if count > count_thresh:
            total_str = binary_repr(idx, bin_size)

            single_decoded_part = []

            start_idx = 0
            for part_size in partition:
                single_decoded_part.append(
                    int(total_str[start_idx:start_idx + part_size], base=2)
                )
                start_idx += part_size

            res.append(single_decoded_part)

    return res
