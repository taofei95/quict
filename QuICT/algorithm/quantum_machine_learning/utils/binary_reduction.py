import string
import numpy as np
import time
import tensorflow


def is_adjont(binary1 , binary2,half_length):
    if binary1 % (2**half_length) != binary1 % (2**half_length):
        return False
    same_bit_bin = binary1 ^ binary2
    same_bit_str = bin(same_bit_bin)
    count = 0
    for item in same_bit_str:
        if item == '1':
            count += 1
    if count == 1:
        return True
    else:
        return False


def reduce_bin(binary1, binary2,half_length ): # binary1 : int
    bin1_int = binary1
    bin2_int = binary2
    binary1 = bin(binary1)
    binary2 = bin(binary2)
    same_bit_bin = bin2_int ^ bin1_int
    reduced_int = bin1_int if bin1_int > bin2_int else bin2_int

    reduce_position = int(np.log2(same_bit_bin))
    reduce_position -= half_length

    reduced_int -= same_bit_bin
    reduced_int += 2 ** reduce_position


    return reduced_int

def check(color:set,half_length):
    """
    :param color:
    :return:  if color contain two adjoint bin strings return True ,else return False
    """
    for bin1 in color:
        for bin2 in color:
            if is_adjont(bin1, bin2,half_length):
                return True
    return False

def may_adj(color:set):
    """
    :param color:
    :return:  if color contain two adjoint bin strings return True ,else return False
    """
    may_adj_list = []
    for bin1 in color:
        for bin2 in color:
            if is_adjont(bin1, bin2):
                may_adj_list.append([bin1, bin2])
    return may_adj_list


def Binary_reduction(bins,half_length:int): 
    """

        Args:
            bins(np.array):
            half_length(int): the length of the longest of bianary.for a qcircuit:it should larger than number of qubits

    """   
    #half_length = 13
    #bins = np.random.randint(0, 2**half_length, 2**13)
    #bins = np.array([0,8,16,24,32,40,48,56])
    bins = bins*(2**half_length)
    color_set = set()

    for i in range(len(bins)):
        color_set.add(bins[i])
    color_set_new = color_set.copy()
    print(len(color_set))
    reduced_set = set()
    start_time = time.time()
    while(1):
        color_set = color_set_new.copy()
        if not check(color_set_new,half_length):
            break                               
        reduced_set.clear()
        count = 0
        for bin1 in color_set:
            for bin2 in color_set:
                if bin1 in reduced_set:
                    break
                if bin2 in reduced_set:
                    continue
                if is_adjont(bin1,bin2,half_length):
                    reduced_set.add(bin1)
                    reduced_set.add(bin2)
                    color_set_new.add(reduce_bin(bin1, bin2,half_length))
                    reduced_str = reduce_bin(bin1, bin2,half_length)
                    color_set_new.remove(bin1)
                    color_set_new.remove(bin2)
                    count += 1                                                                 
 
                                                                 
    #bins = np.random.randint(0, 2**half_length, 2**13)                                    

    return  color_set_new                                       
                                                                           