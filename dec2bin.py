def get_binary_num_as_string(decimal_num, binary_num_str=''):
    if(decimal_num >= 1):
        binary_num_str += get_binary_num_as_string(decimal_num // 2, binary_num_str)
        binary_num_str += str(decimal_num % 2)
    return binary_num_str

def get_reversed_binary_num_as_string(decimal_num):
    return get_binary_num_as_string(decimal_num)[::-1]