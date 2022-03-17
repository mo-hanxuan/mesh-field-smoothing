"""
function: makeObject  ->  use inp file to make an object constructed by elements
"""

import numpy as np

# ------------------------------------------------------import user-define modules
import sys
sys.path.append('G:\\Python code\\Algorithms\\tryOpenGL')
from Element import C3D8
from Element import Object3D


def cleanData(data, delimiter=' '):
    # make data to be the form that np.loadtxt can execute
    dt = []
    for i in range(len(data)):
        if data[i] == ',':
            dt.append(delimiter)
        else:
            dt.append(data[i])
    dt = [''.join(dt)]
    return dt


def makeObject(job, site='dataFile/'):
    # ----------------------------------read file data
    import re

    # job = 't5x5_dense.inp'
    with open(site + job, 'r') as r:
        fl = r.read()

    # ------------------------------------------------------- extract the string
    xyz = re.findall(r'Node(.+?)Element, type=', fl, re.S)
    xyz = xyz[0][1:-2]

    nodes = np.loadtxt(cleanData(xyz))
    nodes = np.reshape(nodes, [int(len(nodes) / 4), 4])
    nodes = nodes[:, 1:]
    nodes = np.mat(nodes)
    print('nodes = \n{}\n'.format(nodes))
    print('type(nodes) = {}\n'.format(type(nodes)))

    # ------------------------------------------------------- extract the string
    ele = re.findall('Element, type=C3D8R\n(.+?)Nset', fl, re.S)
    ele = ele[0][:-1]

    ele = np.loadtxt(cleanData(ele))
    ele = np.reshape(ele, [int(len(ele) / 9), 9])
    ele = ele[:, 1:]
    ele = np.mat(ele)
    ele = ele.astype(int)
    print('ele = \n{}\n'.format(ele))
    # -----------------------------------------------------------------------------------------------
    eles = []
    for i in range(len(ele[:, 0])):
        eles.append(C3D8(number=i + 1, nodes=ele[i, :].tolist()[0]))
    obj1 = Object3D(eles=eles, nodes=nodes)
    obj1.nodes = nodes

    # obj1.ratio_draw = 1. if nodes.max() < 50. else 1./40.
    # obj1.nodes *= obj1.ratio_draw
    # ----------------------------------------------------------------------------------------------- 读文件

    obj1.getFaceEdge()    # get surfaces and corresponding edges

    return obj1