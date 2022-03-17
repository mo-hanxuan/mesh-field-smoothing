# -*- coding: utf-8 -*-

# -------------------------------------------
# Module
# C3D8 单元类,  物体类
# -------------------------------------------

class C3D8(object):
    def __init__(self, number=1, nodes=[]):
        self._n = number  # element number
        self._nodes = nodes
        self._faces = []
        self._edges = []

        if len(self._nodes) != 8:
            raise ValueError('C3D8 element should have 8 nodes !!!')
        if type(self._nodes[0]) != type(1):
            raise ValueError('nodes number of C3D8 should be integer !!!')

        #     v4----- v5
        #    /|      /|
        #   v0------v1|
        #   | |     | |
        #   | v7----|-v6
        #   |/      |/
        #   v3------v2

        faces = [[0, 1, 2, 3],
                 [4, 5, 6, 7],
                 [1, 2, 6, 5],
                 [0, 3, 7, 4],
                 [0, 1, 5, 4],
                 [3, 2, 6, 7]]
        for i in range(len(faces)):
            tem = []
            for xx in faces[i]:
                tem.append(self._nodes[xx])
            edge = [{self._nodes[faces[i][0]], self._nodes[faces[i][1]]},
                    {self._nodes[faces[i][1]], self._nodes[faces[i][2]]},
                    {self._nodes[faces[i][2]], self._nodes[faces[i][3]]},
                    {self._nodes[faces[i][3]], self._nodes[faces[i][0]]}]
            self._faces.append({'nodes': tem, 'edges': edge})

        edge = [[0, 1], [4, 5], [7, 6], [3, 2],
                [0, 3], [1, 2], [5, 6], [4, 7],
                [2, 6], [1, 5], [0, 4], [3, 7]]
        for i in range(len(edge)):
            self._edges.append({self._nodes[edge[i][0]],
                                self._nodes[edge[i][1]]})

class Object3D(object):
    def __init__(self, name='object1', eles=[]):
        self._name = name
        self._eles = eles
        if type(eles) != type([]):
            raise ValueError('input should be a list of element objects')
        if type(eles[0]) != type(C3D8(nodes=[0 for i in range(8)])):
            raise ValueError('input list should composed by class C3D8')
        self.faceSet = []
        self.edgeSet = []
        self.surfaces = []
        self.surfaceEdges = []

    def getFaceEdge(self):
        self.faceSet = []
        for ele in self._eles:
            for face1 in ele._faces:
                # see if the face the same with the previous face
                flag = 1
                for face2 in self.faceSet:
                    if set(face1['nodes']) == set(face2['nodes']):
                        flag = 0
                        face2['ele'].append(ele._n)
                        break
                if flag == 1:
                    self.faceSet.append({'nodes': face1['nodes'],
                                         'edges': face1['edges'],
                                         'ele': [ele._n]})

        self.edgeSet = []
        for ele in self._eles:
            for edge1 in ele._edges:
                # see if the edge the same with the previous edge
                flag = 1
                for edge2 in self.edgeSet:
                    if edge1 == edge2:
                        flag = 0
                        break
                if flag == 1:
                    self.edgeSet.append(edge1)

        self.surfaces = []
        for face in self.faceSet:
            if len(face['ele']) == 1:
                self.surfaces.append(face)

        self.surfaceEdges = []
        for face in self.surfaces:
            for edge1 in face['edges']:
                # see if the edge the same with the previous edge
                flag = 1
                for edge2 in self.surfaceEdges:
                    if edge1 == edge2:
                        flag = 0
                        break
                if flag == 1:
                    self.surfaceEdges.append(edge1)



if __name__ == "__main__":
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
    import numpy as np

    def cleanData(data, delimiter=' '):
        # make data to be the form that np.loadtxt can execute
        dt = []
        for i in range(len(data)):
            if data[i] == ',':
                dt.append(delimiter)
            elif data[i] == '\n':
                dt.append(delimiter)
            else:
                dt.append(data[i])
        dt = [''.join(dt)]
        return dt

    # ----------------------------------read file data
    import re
    import torch
    with open('dataFile\\t5x5.inp', 'r') as r:
        fl = r.read()
    # print(fl)

    # ------------------------------------------------------- extract the string
    xyz = re.findall(r'Node(.+?)Element, type=', fl, re.S)
    xyz = xyz[0][1:-2]

    nodes = np.loadtxt(cleanData(xyz))
    nodes = np.reshape(nodes, [int(len(nodes) / 4), 4])
    nodes = nodes[:, 1:]
    nodes = np.mat(nodes)
    nodes /= 40.
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
    #ele = ele.tolist()
    print('ele = \n{}\n'.format(ele))

    eles = []
    for i in range(len(ele[:, 0])):
        eles.append(C3D8(number=i+1, nodes=ele[i, :].tolist()[0]))
    print('len(eles) = ', len(eles))

    print(eles[5]._faces)

    print(type(eles[0]))

    obj1 = Object3D(eles=eles)
    obj1.getFaceEdge()
    print(len(obj1.faceSet))
    print(len(obj1.edgeSet))
    print(len(obj1.surfaces))
    print(len(obj1.surfaceEdges))


