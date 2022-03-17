from collections import OrderedDict
import numpy as np


def read_txt(file, sep=' ', dtype=float):
    """
        read the txt file, get the data
        data is arrange by columes,
        where the first line shows the name of each colume

        you can specify the dtype of data by arguments dtype,
        but all columes will have same dtype

        return an OrderDict, where each key points to a numpy array
    """
    data = OrderedDict()

    with open(file, 'r') as file:
        for i, line in enumerate(file):
            if i == 0:
                tmp = line.split(sep)
                tmp = tmp[:-1]
                for s in tmp:
                    data[s] = []
            else:
                tmp = line.split(sep)
                tmp = tmp[:-1]
                tmp = list(map(dtype, tmp))
                for j, key in enumerate(data):
                    data[key].append(tmp[j])
    
    for key in data:
        data[key] = np.array(data[key])
    return data


if __name__ == '__main__':

    file = r'D:/BaiduNetdiskWorkspace/server/1026/neuralF/twFit_normalize/nuRate_0.2_mo_0.3/nuCrss_60/Tmax_0.005/Outcome/data/SDV210_fullField.txt'

    data = read_txt(file, sep=' ')

    print(data)
    for key in data:
        print('data[{}].sum() = {}'.format(
            key, data[key].sum()
        ))

