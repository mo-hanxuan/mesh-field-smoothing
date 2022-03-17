# -*- coding: utf-8 -*-

# -------------------------------------------
# Module
# C2D4 单元类
# C3D8 单元类,  物体类
# -------------------------------------------

import numpy as np


class C2D4(object):
    def __init__(self, xy=[[-1., -1.],
                           [1., -1.],
                           [1., 1.],
                           [-1., 1.]]):

        self.shapeF = None  # shape function

        """
        if use self.getPatch(n), you get self.patches and self.outFrame
            self.patches
                type: np.ndarray
                shape: (n*n, 4, 3)
                    1st index enumerates each patch
                    2nd index enumerates the nodes of the patch
                    3rd index enumerates the global coordinates
            self.outFrame
                type: np.ndarray
                shape: (4*n, 2, 3)
                    1st index enumerates each line segment
                    2nd index enumerates two endpoints
                    3rd index enumerates the global coordinates
        """
        self.patches = None  # densified grids inside the quadrangle
        self.outFrame = None  # out Frame of self.patches

        """------------------------------------"""
        # the global coordinate of 4 nodes
        # coordinate can be either 3D or 2D
        # 2D is converted to 3D by setting z = 0 as default
        #
        # the nodes should arrange as following:
        #                    (either list or np.mat is ok)
        #
        #   n3---------n2       [[x0, y0, z0],
        #   |          |         [x1, y1, z1],
        #   |          |         [x2, y2, z2],
        #   |          |         [x3, y3, z3]]
        #   n0---------n1
        """------------------------------------"""
        if type(xy) != type(np.mat([])):
            xy = np.mat(xy)
        if xy.shape == (4, 3):
            self.xy = xy
        elif xy.shape == (4, 2):
            xy = np.append(xy, np.zeros((4, 1)), axis=1)
            self.xy = xy
        else:
            raise ValueError('in C2D4, the nodes coordinate "xy" '
                             'should be of shape (4, 3) or (4, 2)')

    def shapeFun(self, coord=[]):
        """
        the shape function of a quadrangle
        :param coord: natural coordinate [ξ, η] of the given point
                      which has the range of [-1 ~ 1, -1 ~ 1]

        :return: [n0, n1, n2, n3], the shape function values
                                   for 4 nodes respectively
        """
        if type(coord) == type([]):
            pass
        elif type(coord) == type(np.mat([])):
            coord = coord.tolist()
        elif type(coord) == type(np.ndarray((2, 3))):
            coord = coord.tolist()
        else:
            raise ValueError('input coordinate should be '
                             'a list or np.mat or np.ndarray')
        lenC = len(coord)
        if lenC != 2:
            raise ValueError('the input natural coordinate '
                             'for C2D4 element should be 2D')
        nodes = np.mat([[-1., -1.],
                        [ 1., -1.],
                        [ 1.,  1.],
                        [-1.,  1.]])
        spf = [0., 0., 0., 0.]  # shape function
        for i in range(len(spf)):
            spf[i] = 0.25 \
                     * (1. + nodes[i, 0]*coord[0]) \
                     * (1. + nodes[i, 1]*coord[1])
        return spf

    def densify(self, n):
        """
        densify the grid (fill patches) inside this quadrangle

        :param: n: densify the grid by n x n
            for example, if n = 2,
            the densify operation 2 x 2 looks like this

            n3-----------n2
            |  o  |  o  |
            |-----|-----|
            |  o  |  o  |
            n0----------n1

        :return:
            patches[0:n, 0:n, 0:4, 0:3], of type np.ndarray
                ... 0:n  fix ξ and enumerate η
                ... 0:n  fix η and enumerate ξ
                ... 0:4  four nodes for each patches
                ... 0:3 the global coordinates [x, y, z] for each node

            outFrame[0:4, 0:n, 0:2, 0:3], the line segregates of
                                       outer frame of the quadrangle
                ... 0:4  enumerate 4 lines of this outer frame
                ... 0:n  the line segregation
                ... 0:2  two nodes at line segregation endpoints
                ... 0:3  the global coordinates [x, y, z] for each node
        """
        natC = np.zeros((n+1, n+1, 2))  # natural coordinates of patches nodes
        for i in range(len(natC[:, 0, 0])):
            for j in range(len(natC[0, :, 0])):
                natC[i, j, 0] = -1. + i * (2. / n)
                natC[i, j, 1] = -1. + j * (2. / n)
        # print('natC =\n', natC)

        gloC = np.zeros((n+1, n+1, 3))  # global coordinates of patches nodes
        for i in range(len(natC[:, 0, 0])):
            for j in range(len(natC[0, :, 0])):
                spf = self.shapeFun(coord=natC[i, j, :])  # shape function
                # print('natC[i, j, :] =', natC[i, j, :])
                # print('spf =', spf)
                for k in range(len(gloC[i, j, :])):
                    gloC[i, j, k] = 0.
                    for nod in range(4):
                        gloC[i, j, k] += spf[nod] * self.xy[nod, k]
        # print('gloC =\n', gloC)

        patches = np.zeros((n, n, 4, 3))
        for i in range(n):
            for j in range(n):
                patches[i, j, 0, :] = gloC[i, j, :]
                patches[i, j, 1, :] = gloC[i+1, j, :]
                patches[i, j, 2, :] = gloC[i+1, j+1, :]
                patches[i, j, 3, :] = gloC[i, j+1, :]
        # print('patches =', patches)

        outFrame = np.zeros((4, n, 2, 3))
        for j in range(n):
            outFrame[0, j, 0, :] = gloC[j, 0, :]
            outFrame[0, j, 1, :] = gloC[j+1, 0, :]
        for j in range(n):
            outFrame[1, j, 0, :] = gloC[n, j, :]
            outFrame[1, j, 1, :] = gloC[n, j + 1, :]
        for j in range(n):
            outFrame[2, j, 0, :] = gloC[j, n, :]
            outFrame[2, j, 1, :] = gloC[j + 1, n, :]
        for j in range(n):
            outFrame[3, j, 0, :] = gloC[0, j, :]
            outFrame[3, j, 1, :] = gloC[0, j + 1, :]

        return patches, outFrame

    def getPatch(self, n):
        self.patches, self.outFrame = self.densify(n)
        self.patches = self.patches.reshape((n*n, 4, 3))
        self.outFrame = self.outFrame.reshape((4*n, 2, 3))
        """ test if the reshape works
                # a = np.array([[[1., 2.],
                #        [3., 4.]],
                #       [[5., 6],
                #        [7., 8.]]])
                # a = a.reshape((4, 2))
                # print('a =\n', a)
        """


class C3D8(object):
    def __init__(self, number=1, nodes=[],
                 xyz=[]):
        self._n = number  # element number
        self._nodes = nodes
        self._faces = []
        self._edges = []
        self.xyz = xyz

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

    def center(self):  # the coordinates of center point of an element
        if len(self.xyz) != 8:
            raise ValueError('[x, y, z] of 8 nodes '
                             'should be given to the C3D8 element')
        if type(self.xyz) != type(np.mat([])):
            raise ValueError('element.xyz should be of type numpy.mat')
        # print('self.xyz =\n', self.xyz)
        cen = [0. for i in range(self.xyz.shape[1])]
        for i in range(len(cen)):
            cen[i] = self.xyz[:, i].sum() / self.xyz[:, i].shape[0]
        return cen


class Object3D(object):
    def __init__(self, name='object1', eles=[], nodes=[]):
        self._name = name
        self._eles = eles
        if type(eles) != type([]):
            raise ValueError('input should be a list of element objects')
        if type(eles[0]) != type(C3D8(nodes=[0 for i in range(8)])):
            raise ValueError('input list should composed by class C3D8')

        self.nodes = nodes
        if type(nodes) == type(np.mat([])):
            self.nodes = nodes
            if len(nodes[0, :].tolist()[0]) != 3:
                # print('nodes[0, :].tolist() =', nodes[0, :].tolist())
                raise ValueError('nodes coordinates should be 3 dimensional')
        elif type(nodes) == type([]):
            self.nodes = np.mat(nodes)
            if len(nodes[0]) != 3:
                raise ValueError('nodes coordinates should be 3 dimensional')
        else:
            raise ValueError('nodes coordinates should be '
                             'of type list or type matrix')

        self.nod_ele = [set() for i in range(len(self.nodes))]  # from node number to element number
        self.eleNear = [set() for i in range(len(self._eles))]  # get the neighbour element of given element
                                                                # the elements that share common nodes with the given element

        self.faceSet = []
        self.edgeSet = []
        self.surfaces = []
        self.surfaceEdges = []

        # ---------------------------- give [x, y, z] of nodes for each element
        if len(self._eles[0].xyz) == 0:
            for ele in self._eles:
                ele.xyz = np.mat([0., 0., 0.])
                for node in ele._nodes:
                    ele.xyz = np.append(ele.xyz, self.nodes[node-1, :], axis=0)
                ele.xyz = ele.xyz[1:, :]

    def get_nod_ele(self):
        for ele in self._eles:
            for node in ele._nodes:
                self.nod_ele[node - 1].add(ele._n)

    def get_eleNear(self):
        if len(self.nod_ele[0]) == 0:
            self.get_nod_ele()
        for ele in self._eles:
            for node in ele._nodes:
                for ele2 in self.nod_ele[node - 1]:
                    if ele2 != ele._n:
                        self.eleNear[ele].add(ele2)

    def getFaceEdge(self):
        # ---------- initialize
        if len(self.eleNear[0]) == 0:
            self.get_eleNear()
        for ele in self._eles:
            ele.surfaceGlobal = []  # elements' global surface of the object

        self.faceSet = []
        for ele in self._eles:
            for face1 in ele._faces:
                # see if the face the same with the previous face
                flag = 1
                for ele2 in self.eleNear[ele]:
                    if ele2 < ele._n:
                        for face2 in self._eles[ele2 - 1]._faces:
                            if set(face1['nodes']) == set(face2['nodes']):
                                flag = 0
                                break
                if flag == 1:
                    self.faceSet.append({'nodes': face1['nodes'],
                                         'edges': face1['edges'],
                                         'ele': [ele._n]})
                    # find another element that shares this same face
                    flag = 1
                    for ele2 in self.eleNear[ele]:
                        if ele2 > ele._n:
                            for face2 in self._eles[ele2 - 1]._faces:
                                if set(face1['nodes']) == set(face2['nodes']):
                                    self.faceSet[-1]['ele'].append(ele2)
                                    flag = 0
                                    break
                            if flag == 0:
                                break

        self.edgeSet = []
        for ele in self._eles:
            for edge1 in ele._edges:
                # see if the edge the same with the previous edge
                flag = 1
                for ele2 in self.eleNear[ele]:
                    if ele2 < ele._n:
                        for edge2 in self._eles[ele2 - 1]._edges:
                            if edge1 == edge2:
                                flag = 0
                                break
                if flag == 1:
                    self.edgeSet.append(edge1)

        self.surfaces = []
        for face in self.faceSet:
            if len(face['ele']) == 1:
                self.surfaces.append(face)
                self._eles[face['ele'][0] - 1].surfaceGlobal\
                    .append({'face': face,
                             'number': len(self.surfaces)-1})

        self.surfaceEdges = []
        for i, face in enumerate(self.surfaces):
            ele1 = face['ele'][0]
            eNear = list(self.eleNear[ele1 - 1])
            eNear.append(ele1)  # eNear is neighbor element, include element itself
            # print('eNear =', eNear)
            for edge1 in face['edges']:
                # see if the edge the same with the previous edge
                flag = 1
                for ele2 in eNear:
                    for face2 in self._eles[ele2 - 1].surfaceGlobal:
                        if face2['number'] < i:  # previous surface
                            for edge2 in face2['face']['edges']:
                                if edge1 == edge2:
                                    flag = 0
                                    break
                        if flag == 0:
                            break
                    if flag == 0:
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

    obj1 = Object3D(eles=eles, nodes=nodes)
    obj1.getFaceEdge()
    print(len(obj1.faceSet))
    print(len(obj1.edgeSet))
    print(len(obj1.surfaces))
    print(len(obj1.surfaceEdges))


