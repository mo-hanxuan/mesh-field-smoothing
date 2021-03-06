"""
    get the body that composed by C3D8 elements

    attention: node index start from 1, not 0!
    (注意！ 本程序的节点编号从 1 开始， 与 inp 中的节点编号相同)

    hashNodes algorithm:
        very fast to identify all facets,
        facets stored by a dict(), 
        (see 'facetDic')
        key of dict:
            sorted tuple of node numbers of a facet


    加速方法：
        self.eleNeighbor 用 字典
        eleFacet 用 字典

    
    author: mohanxuan, mo-hanxuan@sjtu.edu.cn
    license: Apache licese
"""
from typing import ValuesView
import torch as tch
import threading, time

from torch._C import Value
from progressBar import *

from functools import cmp_to_key


def leftIdx(lis, idx):
    left = idx - 1
    if left < 0:
        left = len(lis) - 1
    return left


def rightIdx(lis, idx):
    right = idx + 1
    if right > len(lis) - 1:
        right = 0
    return right


def sortByClockwise(lis):
    """
        sort a list by clockwise or counterClockwise order
        applied for the facet nodes list

        start from the smallest nodes,
        if original direction's next node < reverse direction's next node
            use original direction
        else:
            use reverse direction
    """
    ### find the smallest node's idx
    start = lis.index(min(lis))
    res = [lis[start]]
    if lis[rightIdx(lis, start)] < lis[leftIdx(lis, start)]:
        cur = start
        for i in range(len(lis) - 1):
            res.append(lis[rightIdx(lis, cur)])
            cur = rightIdx(lis, cur)
    else:
        cur = start
        for i in range(len(lis) - 1):
            res.append(lis[leftIdx(lis, cur)]) 
            cur = leftIdx(lis, cur)
    return res


def readInp(fileName='donut.inp'):
    """
    read the inp file, returns:
        nodes: the coordinates of all nodes
        elements: corresponding node numbers of all elements
    """

    nodes = {}
    cout = False
    with open(fileName, 'r') as file:
        for line in file:
            if '*' in line:
                if cout:
                    break
            
            if cout:
                data = line.split(',')
                data = list(map(float, data))
                nodes[int(data[0])] = data[1:]
            
            if '*Node' in line or '*NODE' in line or '*node' in line:
                cout = True

    elements = tch.tensor([], dtype=tch.int)
    cout = False
    text = []
    with open(fileName, 'r') as file:
        for line in file:
            if '*' in line:
                if cout:
                    break
            
            if cout:
                data = line[:-1].rstrip().rstrip(',')
                data = data.split(',')
                tex = []
                for x in data:
                    tex.append(x)
                text.extend(tex)
            
            if '*ELEMENT' in line or '*Element' in line or '*element' in line:
                cout = True
    data = list(map(int, text))
    elements = tch.tensor(data)
    elements = elements.reshape((-1, 9))
    elements = elements[:, 1:]

    return nodes, elements


def readDataFrame(fileName):
    with open(fileName, "r") as file:
        i = -1
        for line in file:
            i += 1
            if i == 0:  # the first line
                keys = line.split()
                data = {key:[] for key in keys}
            else:  # the 2nd -> n lines
                tmp = list(map(float, line.split()))
                for idx, key in enumerate(keys):
                    data[key].append(tmp[idx])
    return data


class ElementsBody(object):
    """
        author: mohanxuan,  mo-hanxuan@sjtu.edu.cn
        license: Apache license
    """
    def __init__(self, nodes, elements, name='elesBody1'):
        """
        nodes[] are coordinates of all nodes
        elements[] are the corresponding node number for each element 
        """
        for node in nodes:
            if (not tch.is_tensor(nodes[node])) and type(nodes[node]) != type([]):
                print('type(nodes[node]) =', type(nodes[node]))
                raise ValueError('item in nodes should be a torch tensor or a list')
            break
        if not tch.is_tensor(elements):
            raise ValueError('elements should be a torch tensor')
        for node in nodes:
            if tch.is_tensor(nodes[node]):
                print('nodes[node].size() =', nodes[node].size())
                if nodes[node].size()[0] != 3:
                    raise ValueError('nodes coordinates should 3d')
            else:
                # print('len(nodes[node]) =', len(nodes[node]))
                if len(nodes[node]) != 3:
                    raise ValueError('nodes coordinates should 3d')
            break
        if len(elements.size()) != 2:
            raise ValueError('len(elements.size()) should be 2 !')
        if elements.max() != max(nodes):
            print('elements.max() - 1 =', elements.max() - 1)
            print('max(nodes) =', max(nodes))
            raise ValueError('maximum element nodes number not consistent with nodes number')
        
        self.nodes = nodes
        self.elements = elements
        self.name = name

        self.nod_ele = None
        self.eleNeighbor = None
        self.allFacets = None  # all element facets of this body
        self.eCen = {}  # elements center
        self._celent = {}  # chosen element length
        self._volumes = {}

        # node number starts from 1
        # element number starts from 0

        """
                      v3------v7
                     /|      /|
                    v0------v4|
                    | |     | |
                    | v2----|-v6
             y ^    |/      |/
               |    v1------v5
               --->
              /    x
             z       
        """
    
    def eles(self):
        # element nodes coordinates: self.eles
        print('now, we plug in the node coordinates for all elements')
        eles = tch.tensor([])
        for i, ele in enumerate(self.elements):
            
            if i % 100 == 0:
                percent = i / self.elements.size()[0] * 100.
                progressBar_percentage(percent)

            for j in range(len(ele)):
                eles = tch.cat((eles, tch.tensor(self.nodes[ele[j]])))
        
        eles = eles.reshape((-1, 8, 3))
        self.eles = eles
        return self.eles


    def get_nod_ele(self):  # node number -> element number
        if not self.nod_ele:
            nod_ele = {i:set() for i in self.nodes}
            for iele, ele in enumerate(self.elements):
                for node in ele:
                    nod_ele[int(node)].add(iele)
            self.nod_ele = nod_ele

        return self.nod_ele


    def get_eleNeighbor(self):  # get the neighbor elements of the given element
        if not self.nod_ele:
            self.get_nod_ele()
        if not self.eleNeighbor:
            neighbor = {i: set() for i in range(len(self.elements))}
            for iele, ele in enumerate(self.elements):
                for node in ele:
                    for eNei in self.nod_ele[int(node)]:
                        if eNei != iele:
                            neighbor[iele].add(eNei)
            self.eleNeighbor = neighbor
        return self.eleNeighbor
    
    
    def get_allFacets(self):  # get the element facets
        """
            'HashNodes algorithm': (invented by MoHanxuan)
                a very fast algorithm for facet generating!
                use hash dict to store all facets
                nodes number is the key to identify a facet
                e.g.:
                    facet with nodes [33, 12, 5, 7] has key '5,7,12,33'
                    , where 4 nodes are transverted to 
                    a key of tuple with sorted sequence

                see variable 'facetDic'
        """

        if not hasattr(self, "facetDic"):
            
            # facet normal points to the positive direction of natural coordinates
            facets = tch.tensor([[0, 1, 2, 3],  # x
                                 [4, 5, 6, 7], 
                                
                                 [1, 5, 6, 2],  # y
                                 [0, 4, 7, 3],
                                
                                 [3, 2, 6, 7],  # z
                                 [0, 1, 5, 4]], dtype=tch.int)
            
            facetDic = {}
            eleFacet = {
                i: [[], [], []] for i in range(len(self.elements))
            }

            print('now, generate all the element facets')
            timeBeg = time.time()
            facetDic = {}
            for iele, ele in enumerate(self.elements):

                if iele % 100 == 0:
                    percentage = iele / len(self.elements) * 100.
                    progressBar_percentage(percentage)

                for ifacet, facet in enumerate(facets):  # 6 facets
                    f = []
                    for node in facet:
                        f.append(int(ele[node]))
                    tmp = tuple(sortByClockwise(f))
                    if tmp in facetDic:
                        facetDic[tmp].append(iele)
                    else:
                        facetDic[tmp] = [iele]
                    eleFacet[iele][ifacet // 2].append(tmp)
            
            print('')  # break line for progress bar
            print('\033[33m{}\033[0m {} \033[35m{}\033[0m'.format(
                'time for facetDic is', time.time() - timeBeg, "seconds"
            ))

            print('\033[33m{}\033[0m {} \033[35m{}\033[0m'.format(
                'consuming time for facets generating is', time.time() - timeBeg, "seconds"
            ))

            print('\033[35m{}\033[0m {} \033[35m{}\033[0m'.format(
                'There are', len(facetDic), 'facets'
            ))
            self.facetDic = facetDic
            self.eleFacet = eleFacet
        
        return self.facetDic   


    def get_allFacets(self):  # get the element facets
        """
            'HashNodes algorithm': (invented by MoHanxuan)
                a very fast algorithm for facet generating!
                use hash dict to store all facets
                nodes number is the key to identify a facet
                e.g.:
                    facet with nodes [33, 12, 5, 7] has key '5,7,12,33'
                    , where 4 nodes are transverted to 
                    a key of tuple with sorted sequence

                see variable 'facetDic'
        """

        if not hasattr(self, "facetDic"):
            
            # facet normal points to the positive direction of natural coordinates
            facets = tch.tensor([[0, 1, 2, 3],  # x
                                 [4, 5, 6, 7], 
                                
                                 [1, 5, 6, 2],  # y
                                 [0, 4, 7, 3],
                                
                                 [3, 2, 6, 7],  # z
                                 [0, 1, 5, 4]], dtype=tch.int)
            
            facetDic = {}
            eleFacet = {
                i: [[], [], []] for i in range(len(self.elements))
            }

            print('now, generate all the element facets')
            timeBeg = time.time()
            facetDic = {}
            for iele, ele in enumerate(self.elements):

                if iele % 100 == 0:
                    percentage = iele / len(self.elements) * 100.
                    progressBar_percentage(percentage)

                for ifacet, facet in enumerate(facets):  # 6 facets
                    f = []
                    for node in facet:
                        f.append(int(ele[node]))
                    tmp = tuple(sortByClockwise(f))
                    if tmp in facetDic:
                        facetDic[tmp].append(iele)
                    else:
                        facetDic[tmp] = [iele]
                    eleFacet[iele][ifacet // 2].append(tmp)
            
            print('')  # break line for progress bar
            print('\033[33m{}\033[0m {} \033[35m{}\033[0m'.format(
                'time for facetDic is', time.time() - timeBeg, "seconds"
            ))

            print('\033[33m{}\033[0m {} \033[35m{}\033[0m'.format(
                'consuming time for facets generating is', time.time() - timeBeg, "seconds"
            ))

            print('\033[35m{}\033[0m {} \033[35m{}\033[0m'.format(
                'There are', len(facetDic), 'facets'
            ))
            self.facetDic = facetDic
            self.eleFacet = eleFacet
        
        return self.facetDic 


    def get_outerFacetDic(self):  # get the element facets
        """
            'HashNodes algorithm': (invented by MoHanxuan)
                a very fast algorithm for facet generating!
                use hash dict to store all facets
                nodes number is the key to identify a facet
                e.g.:
                    facet with nodes [33, 12, 5, 7] has key '5,7,12,33'
                    , where 4 nodes are transverted to 
                    a key of tuple with sorted sequence
        """

        if not hasattr(self, "outerFacetDic"):
            
            # facet normal points to the positive direction of natural coordinates
            facets = tch.tensor([[0, 1, 2, 3],  # x
                                 [4, 5, 6, 7], 
                                
                                 [1, 5, 6, 2],  # y
                                 [0, 4, 7, 3],
                                
                                 [3, 2, 6, 7],  # z
                                 [0, 1, 5, 4]], dtype=tch.int)
            
            facetDic = {}

            print('now, generate all the element facets')
            timeBeg = time.time()
            facetDic = {}
            for iele, ele in enumerate(self.elements):

                if iele % 100 == 0:
                    percentage = iele / len(self.elements) * 100.
                    progressBar_percentage(percentage)

                for ifacet, facet in enumerate(facets):  # 6 facets
                    f = []
                    for node in facet:
                        f.append(int(ele[node]))
                    tmp = tuple(sortByClockwise(f))
                    if tmp in facetDic:  # facet shared by 2 elements, inner facet, delete it
                        del facetDic[tmp]
                    else:
                        facetDic[tmp] = iele
            
            print('')  # break line for progress bar
            print('\033[33m{}\033[0m {} \033[35m{}\033[0m'.format(
                'time for OuterFacetDic is', time.time() - timeBeg, "seconds"
            ))

            print('\033[33m{}\033[0m {} \033[35m{}\033[0m'.format(
                'consuming time for outer facets generating is', time.time() - timeBeg, "seconds"
            ))

            print('\033[35m{}\033[0m {} \033[35m{}\033[0m'.format(
                'There are', len(facetDic), 'facets'
            ))
            self.outerFacetDic = facetDic
        
        return self.outerFacetDic 


    def get_facialEdgeDic(self):  # get the element edges
        """
            get all facial edges (no overlapped) of this body
            use hash table, or dict,
                key: tuple of sorted nodes
                value: edge number
            
            author: mo-hanxuan@sjtu.edu.cn
            license: Apache 2.0
        """
        if not hasattr(self, "outerFacetDic"):
            self.get_outerFacets()

        if not hasattr(self, "facialEdgeDic"):
            
            # edges points to the positive direction of natural coordinates
            edges = tch.tensor([
                [0, 1], [1, 2], [2, 3], [3, 0]                   
            ], dtype=tch.int)

            print('now, generate all the facial edges')
            timeBeg = time.time()
            edgeDic = {}  # sorted tuple of edge nodes
            for facet in self.outerFacetDic:
                for edge in edges:
                    f = [facet[edge[0]], facet[edge[1]]]
                    tmp = tuple(sorted(f))
                    if tmp in edgeDic:
                        edgeDic[tmp].append(facet)
                    else:
                        edgeDic[tmp] = [facet]
            
            print('\033[35m{}\033[0m {} \033[35m{}\033[0m'.format(
                'There are', len(edgeDic), 'facial edges'
            ))
            self.facialEdgeDic = edgeDic
        
        return self.facialEdgeDic


    def getVolumes(self, eleNum='all'):
        """
            compute the volume of each element
        """
        ncNode = tch.tensor([
            [-1,  1,  1], 
            [-1, -1,  1], 
            [-1, -1, -1], 
            [-1,  1, -1], 
            [ 1,  1,  1], 
            [ 1, -1,  1], 
            [ 1, -1, -1], 
            [ 1,  1, -1],
        ], dtype=tch.int)

        volumes = []

        if eleNum == 'all':
            print('\n now we begin to compute the volume of each element')
            for iele, ele in enumerate(self.elements):
                if iele % 100 == 0:
                    progressBar_percentage((iele / len(self.elements)) * 100.)
                
                eleCoor = tch.tensor([])
                for node in ele:
                    node = node.tolist()
                    eleCoor = tch.cat((eleCoor, tch.tensor(self.nodes[node])), dim=0)
                eleCoor = eleCoor.reshape((-1, 3))

                jacobi = tch.tensor([])
                ksi = tch.tensor([0., 0., 0.], requires_grad=True)
                coo = tch.tensor([0., 0., 0.])

                for dm in range(3):
                    tem = 0.125  # shape function
                    for dm1 in range(len(ncNode[0, :])):
                        tem1 = ncNode[:, dm1] * ksi[dm1]
                        tem1 = 1. + tem1
                        tem *= tem1
                    coo[dm] = (tem * eleCoor[:, dm]).sum()
                    tuple_ = tch.autograd.grad(coo[dm], ksi, retain_graph=True)
                    jacobi = tch.cat((jacobi, tuple_[0]))
                
                jacobi = jacobi.reshape((-1, 3))
                # if iele < 5:
                #     print('jacobi =\n', jacobi)
                #     print('tch.det(jacobi) =', tch.det(jacobi))
                volumes.append((tch.det(jacobi) * 8.).tolist())
            
            print('\n')  # line break for the progress bar
            self.volumes = tch.tensor(volumes)
            return self.volumes
        else:
            if eleNum in self._volumes:
                return self._volumes[eleNum]
            else:
                iele = int(eleNum)
                ele = self.elements[iele]
                eleCoor = tch.tensor([])
                for node in ele:
                    node = node.tolist()
                    eleCoor = tch.cat((eleCoor, tch.tensor(self.nodes[node])), dim=0)
                eleCoor = eleCoor.reshape((-1, 3))

                jacobi = tch.tensor([])
                ksi = tch.tensor([0., 0., 0.], requires_grad=True)
                coo = tch.tensor([0., 0., 0.])

                for dm in range(3):
                    tem = 0.125  # shape function
                    for dm1 in range(len(ncNode[0, :])):
                        tem1 = ncNode[:, dm1] * ksi[dm1]
                        tem1 = 1. + tem1
                        tem *= tem1
                    coo[dm] = (tem * eleCoor[:, dm]).sum()
                    tuple_ = tch.autograd.grad(coo[dm], ksi, retain_graph=True)
                    jacobi = tch.cat((jacobi, tuple_[0]))
                
                jacobi = jacobi.reshape((-1, 3))
                self._volumes[eleNum] = (tch.det(jacobi) * 8.).tolist()
                return self._volumes[eleNum]
    
    
    def get_eLen(self, mod="first"):
        """
            get the characteristic element length
        """
        if not hasattr(self, 'eLen'):
            if mod == "first":
                ### first, get the average element volume
                if not hasattr(self, 'volumes'):
                    vol = self.getVolumes(eleNum=0)
                print('\033[31m{}\033[0m \033[33m{}\033[0m'.format('volume (ele No.0) =', vol))
                self.eLen = vol ** (1./3.)
                print('\033[31m{}\033[0m \033[33m{}\033[0m'.format('characteristic element length (No.0) =', self.eLen))
            
            elif mod == "average":
                if not hasattr(self, 'volumes'):
                    vol = self.getVolumes(eleNum="all")
                self.eLen = (vol.sum() / len(vol)) ** (1./3.)
                print('\033[31m{}\033[0m \033[33m{}\033[0m'.format(
                    'characteristic element length (average) =', self.eLen
                ))
            
            else:
                print("\033[31;1m{}\033[0m".format(
                    "error in function get_eLen(), args 'mod' can only be 'first' or 'average'"
                ))
        return self.eLen
    

    def celent(self, iele):
        """
            chosen element length
            input the id of element (iele), 
            get the characteristic element length of this element
        """
        if iele in self._celent:
            return self._celent[iele]
        else:
            vol = self.getVolumes(eleNum=iele)
            self._celent[iele] = vol ** (1./3.)
            return self._celent[iele]
    

    def getFaceNode(self):
        if not hasattr(self, 'allFacets'):
            self.get_allFacets()
        elif self.allFacets == None:
            self.get_allFacets()
        faceNode = set()
        for facet in range(len(self.allFacets['ele'])):
            if len(self.allFacets['ele'][facet]) == 1:
                faceNode |= set(self.allFacets['node'][facet])
        self.faceNode = faceNode
        return faceNode
    

    def getXYZface(self):
        """
            get the X, Y, Z surfaces for PBC (periodic boundary condition)
            some future improments:
                for parallelogram (平行四边形) or parallel hexahedron(平行六面体)
                we can use face normal to define:
                    x0Face, x1Face, y0Face, y1Face, z0Face, z1Face
        """
        x0Face, x1Face, y0Face, y1Face, z0Face, z1Face = set(), set(), set(), set(), set(), set()
        if not hasattr(self, 'faceNode'):
            self.getFaceNode()
        if not hasattr(self, 'eLen'):
            self.get_eLen()
        
        xMin_key = min(self.nodes, key=lambda x: self.nodes[x][0])
        if xMin_key not in self.faceNode:
            print('xMin_key =', xMin_key)
            raise ValueError('nodes with minimum or maximum coordinates not in outer face set!')
        xMin = self.nodes[xMin_key][0]

        xMax_key = max(self.nodes, key=lambda x: self.nodes[x][0])
        if xMax_key not in self.faceNode:
            raise ValueError('nodes with minimum or maximum coordinates not in outer face set!')
        xMax = self.nodes[xMax_key][0]

        yMin_key = min(self.nodes, key=lambda x: self.nodes[x][1])
        if yMin_key not in self.faceNode:
            raise ValueError('nodes with minimum or maximum coordinates not in outer face set!')
        yMin = self.nodes[yMin_key][1]

        yMax_key = max(self.nodes, key=lambda x: self.nodes[x][1])
        if yMax_key not in self.faceNode:
            raise ValueError('nodes with minimum or maximum coordinates not in outer face set!')
        yMax = self.nodes[yMax_key][1]

        zMin_key = min(self.nodes, key=lambda x: self.nodes[x][2])
        if zMin_key not in self.faceNode:
            raise ValueError('nodes with minimum or maximum coordinates not in outer face set!')
        zMin = self.nodes[zMin_key][2]

        zMax_key = max(self.nodes, key=lambda x: self.nodes[x][2])
        if zMax_key not in self.faceNode:
            raise ValueError('nodes with minimum or maximum coordinates not in outer face set!')
        zMax = self.nodes[zMax_key][2]

        print('xMin = {}, xMax = {}, yMin = {}, yMax = {}, zMin = {}, zMax = {}'.format(
            xMin, xMax, yMin, yMax, zMin, zMax
        ))

        eLen = self.eLen
        for node in self.faceNode:
            if abs(self.nodes[node][0] - xMin) < eLen * 1.e-4:
                x0Face.add(node)
            if abs(self.nodes[node][0] - xMax) < eLen * 1.e-4:
                x1Face.add(node)
            if abs(self.nodes[node][1] - yMin) < eLen * 1.e-4:
                y0Face.add(node)
            if abs(self.nodes[node][1] - yMax) < eLen * 1.e-4:
                y1Face.add(node)
            if abs(self.nodes[node][2] - zMin) < eLen * 1.e-4:
                z0Face.add(node)
            if abs(self.nodes[node][2] - zMax) < eLen * 1.e-4:
                z1Face.add(node)
        self.x0Face, self.x1Face, \
        self.y0Face, self.y1Face, \
        self.z0Face, self.z1Face \
        = \
            x0Face, x1Face, \
            y0Face, y1Face, \
            z0Face, z1Face
        
        print(
            'len(x0Face) = {}, len(x1Face) = {}, \n'
            'len(y0Face) = {}, len(y1Face) = {}, \n'
            'len(z0Face) = {}, len(z1Face) = {},'.format(
                len(x0Face), len(x1Face), len(y0Face), len(y1Face), len(z0Face), len(z1Face),
            )
        )
    

    def getEdgeVertexForPBC(self):
        """
            get the 12 edges and 8 vertexes for PBC (periodic boundary condition)
        """
        if not hasattr(self, 'x0Face'):
            self.getXYZface()
        faces = [
            [self.x0Face, self.x1Face],
            [self.y0Face, self.y1Face],
            [self.z0Face, self.z1Face],
        ]
        permu = [[1, 2], [2, 0], [0, 1]]
        
        xlines, ylines, zlines = [], [], []
        lines = [xlines, ylines, zlines]

        for dm in range(len(faces)):
            edge1 = faces[permu[dm][0]][0] & faces[permu[dm][1]][0]
            edge2 = faces[permu[dm][0]][0] & faces[permu[dm][1]][1]
            edge3 = faces[permu[dm][0]][1] & faces[permu[dm][1]][0]
            edge4 = faces[permu[dm][0]][1] & faces[permu[dm][1]][1]
            lines[dm].extend([edge1, edge2, edge3, edge4])
        
        # get the outer vertexes
        vertexes = set()
        for dm in range(len(lines)):
            for edge1 in lines[dm]:
                for dm2 in permu[dm]:
                    for edge2 in lines[dm2]:
                        vertexes |= (edge1 & edge2)
        self.vertexes = vertexes

        # distinguish the vertexes (very important)
        v_x0y0z0 = list(self.x0Face & self.y0Face & self.z0Face)[0]
        v_x0y0z1 = list(self.x0Face & self.y0Face & self.z1Face)[0]
        v_x0y1z0 = list(self.x0Face & self.y1Face & self.z0Face)[0]
        v_x0y1z1 = list(self.x0Face & self.y1Face & self.z1Face)[0]     
        v_x1y0z0 = list(self.x1Face & self.y0Face & self.z0Face)[0]
        v_x1y0z1 = list(self.x1Face & self.y0Face & self.z1Face)[0]
        v_x1y1z0 = list(self.x1Face & self.y1Face & self.z0Face)[0]
        v_x1y1z1 = list(self.x1Face & self.y1Face & self.z1Face)[0]
        print(
            'v_x0y0z0, v_x0y0z1, v_x0y1z0, v_x0y1z1, v_x1y0z0, v_x1y0z1, v_x1y1z0, v_x1y1z1 =',
            v_x0y0z0, v_x0y0z1, v_x0y1z0, v_x0y1z1, v_x1y0z0, v_x1y0z1, v_x1y1z0, v_x1y1z1
        )
        
        # seperate the lines by beg node, end node, and inside nodes
        x0Set = {v_x0y0z0, v_x0y0z1, v_x0y1z0, v_x0y1z1}
        x1Set = {v_x1y0z0, v_x1y0z1, v_x1y1z0, v_x1y1z1}
        for i_line, line in enumerate(xlines):
            beg = list(x0Set & line)[0]
            end = list(x1Set & line)[0]
            inside = line - {beg} - {end}
            xlines[i_line] = {'beg': beg, 'end': end, 'inside': sorted(inside, key=lambda a: self.nodes[a][0])}
        
        y0Set = {v_x1y0z0, v_x0y0z0, v_x1y0z1, v_x0y0z1}
        y1Set = {v_x1y1z0, v_x0y1z0, v_x1y1z1, v_x0y1z1}
        for i_line, line in enumerate(ylines):
            beg = list(y0Set & line)[0]
            end = list(y1Set & line)[0]
            inside = line - {beg} - {end}
            ylines[i_line] = {'beg': beg, 'end': end, 'inside': sorted(inside, key=lambda a: self.nodes[a][1])}
        
        z0Set = {v_x0y0z0, v_x0y1z0, v_x1y0z0, v_x1y1z0}
        z1Set = {v_x0y0z1, v_x0y1z1, v_x1y0z1, v_x1y1z1}
        for i_line, line in enumerate(zlines):
            beg = list(z0Set & line)[0]
            end = list(z1Set & line)[0]
            inside = line - {beg} - {end}
            zlines[i_line] = {'beg': beg, 'end': end, 'inside': sorted(inside, key=lambda a: self.nodes[a][2])}
        
        print('\033[36m' 'xlines =' '\033[0m')
        for edge in xlines:
            print(edge)
        print('\033[36m' 'ylines =' '\033[0m')
        for edge in ylines:
            print(edge)
        print('\033[36m' 'zlines =' '\033[0m')
        for edge in zlines:
            print(edge)
        
        self.v_x0y0z0, self.v_x0y0z1, self.v_x0y1z0, self.v_x0y1z1, self.v_x1y0z0, self.v_x1y0z1, self.v_x1y1z0, self.v_x1y1z1 = \
            v_x0y0z0, v_x0y0z1, v_x0y1z0, v_x0y1z1, v_x1y0z0, v_x1y0z1, v_x1y1z0, v_x1y1z1
        self.xlines, self.ylines, self.zlines = xlines, ylines, zlines
        return


    def getNodeGraph(self):
        """
            get the node link of the  body
            every node links to other nodes, as a graph
        """
        if not hasattr(self, 'graph'):
            links = [
                [0, 1], [1, 2], [2, 3], [3, 0],
                [4, 5], [5, 6], [6, 7], [7, 4],
                [0, 4], [3, 7], [2, 6], [1, 5]
            ]
            graph = {i: set() for i in self.nodes}
            for ele_ in self.elements:
                ele = ele_.tolist()
                for link in links:
                    graph[ele[link[0]]].add(ele[link[1]])
                    graph[ele[link[1]]].add(ele[link[0]])
            self.graph = graph
        return self.graph
    

    def getFaceForPBC(self):
        """
            get the face-pairs for periodic boundary condition (PBC)
            return:
                faceMatch
        """
        eps = 1.e-3  # a value nearly zero
        tolerance = eps ** 2 * 1000.  # the tolerance for vector difference

        def compare(a, b):
            """
                first compare by idx1,
                if equal, then compare by idx2
            """
            if abs(self.nodes[a][idx1] - self.nodes[b][idx1]) > eLen * eps:
                if self.nodes[a][idx1] > self.nodes[b][idx1]:
                    return 1
                else:
                    return -1
            else:
                if abs(self.nodes[a][idx2] - self.nodes[b][idx2]) > eLen * eps:
                    if self.nodes[a][idx2] > self.nodes[b][idx2]:
                        return 1
                    else:
                        return -1
                else:
                    return 0

        if not hasattr(self, 'x0Face'):
            self.getXYZface()
        if not hasattr(self, 'graph'):
            self.getNodeGraph()
        if not hasattr(self, 'eLen'):
            self.get_eLen()
        faceMatch, baseNodes = [], []
        pairs = [
            [self.x0Face, self.x1Face],
            [self.y0Face, self.y1Face],
            [self.z0Face, self.z1Face]
        ]
        permu = [[1, 2], [2, 0], [0, 1]]
        for idx, pair in enumerate(pairs):
            faceMatch.append({})
            f0, f1 = pair[0], pair[1]
            # find the node with minimum coordinates
            idx1, idx2 = permu[idx][0], permu[idx][1]
            eLen = self.eLen
            n0 = min(f0, key=cmp_to_key(compare))
            n1 = min(f1, key=cmp_to_key(compare))
            baseNodes.append([n0, n1])
            
            faceMatch[-1][n0] = n1
            print('\033[31m' 'n0 = {}, n1 = {}' '\033[0m'.format(n0, n1))
            
            ## start from n0, n1; and traverse other nodes by BFS
            visited0 = {i: False for i in f0}
            print('len(f0) = {}, len(f1) = {}'.format(len(f0), len(f1)))
            if len(f0) != len(f1):
                raise ValueError(
                    '\033[31m' 
                    'nodes quantity does not coincide for opposite faces, f0 ({}) nodes != f1 ({}) nodes' 
                    '\033[0m'.format(len(f0), len(f1))
                )
            lis0, lis1 = [n0], [n1]
            visited0[n0] = True
            while lis0:
                lisNew0, lisNew1 = [], []
                for i_node0, node0 in enumerate(lis0):
                    node1 = lis1[i_node0]

                    if len(self.graph[node0]) != len(self.graph[node1]):
                        print('\033[31m''len(self.graph[node0]) = {} \nlen(self.graph[node1]) = {}''\033[0m'.format(
                            len(self.graph[node0]), len(self.graph[node1])
                        ))

                    vec0s = {}
                    for nex0 in self.graph[node0]:
                        if nex0 in f0:
                            if not visited0[nex0]:
                                visited0[nex0] = True
                                lisNew0.append(nex0)
                                vec0s[nex0] = tch.tensor(self.nodes[nex0]) - tch.tensor(self.nodes[node0])
                    
                    # from another face (f1), find the most similar vec
                    vec1s = {}
                    for nex1 in self.graph[node1]:
                        if nex1 in f1:
                            vec1s[nex1] = tch.tensor(self.nodes[nex1]) - tch.tensor(self.nodes[node1])
                    
                    # link nex0 to nex1
                    for nex0 in vec0s:
                        partner = min(vec1s, key=lambda x: ((vec0s[nex0] - vec1s[x])**2).sum())
                        ## test whether nex0 and partner coincide with each other
                        relativeError = (((vec0s[nex0] - vec1s[partner]) / eLen) ** 2).sum()         
                        if relativeError < tolerance:
                            faceMatch[-1][nex0] = partner
                        else:
                            print(
                                '\033[31m'
                                'node0 = {}, nex0 = {}, node1 = {}, nex1 = {}, \n''vec0 = {}, vec1 = {}'
                                '\033[0m'.format(
                                    node0, nex0, node1, nex1, vec0s[nex0], vec1s[partner],
                                )
                            )
                            print(
                                '\033[33m''warning! relativeError ({:5f}) > tolerance ({}) '
                                'between vector({} --> {}) and vector({} --> {})'
                                '\033[0m'.format(
                                relativeError, tolerance, node0, nex0, node1, nex1
                            ))
                            omit = input('\033[36m' 'do you want to continue? (y/n): ' '\033[0m')
                            if omit == 'y' or omit == '':
                                remainTol = input('\033[36m''remain the current tolerance? (y/n): ' '\033[0m')
                                if remainTol == 'n':
                                    tolerance = float(input('\033[36m''reset tolerance = ' '\033[0m'))
                                faceMatch[-1][nex0] = partner
                            else:
                                raise ValueError('relativeError > tolerance, try to enlarge tolerence instead')
                for nex0 in lisNew0:
                    lisNew1.append(faceMatch[-1][nex0])
                lis0, lis1 = lisNew0, lisNew1
        
        self.faceMatch, self.baseNodes = faceMatch, baseNodes
        return self.faceMatch


    def eleCenter(self, iele):
        if iele in self.eCen:
            return self.eCen[iele]
        else:
            nodes = [self.nodes[int(j)] for j in self.elements[iele]]
            nodes = tch.tensor(nodes)
            self.eCen[iele] = [
                nodes[:, 0].sum() / len(nodes), 
                nodes[:, 1].sum() / len(nodes), 
                nodes[:, 2].sum() / len(nodes), 
            ]
            return self.eCen[iele]


    def findHorizon(self, iele, inHorizon, *args):
        """
            find horizon elements for a given element
                (horizon means a specific range around the given element)
            iuput:
                iele: int, id of the given element 
                inHorizon: function, judge whether ele is inside horizon 
                    (generally by the ele's center xyz)
            output:
                horizon: list, the element number list of the horizon
            methods:
                use BFS to serach neighbor elements layer by layer
                (which prevent searching all elements from whold body)
            note:
                elements id starts from 0
            author: mo-hanxuan@sjtu.edu.cn
            LICENSE: Apache license
        """
        
        ### preprocess of data preparation, 
        ### automatically executes at the 1st call of this function
        ###     inorder to save time for many following calls
        if (not hasattr(self, "eleNeighbor")) or self.eleNeighbor == None:
            self.get_eleNeighbor()
        if (not hasattr(self, "eLen")) or self.eLen == None:
            self.get_eLen(mod="average")
        if len(self.eCen) != len(self.elements):  # prepare self.eCen for all elements
            for idx in range(len(self.elements)):
                self.eleCenter(idx)
        
        horizon = {iele}
        lis = [iele]
        while lis:
            lisNew = []
            for ele in lis:
                for nex in self.eleNeighbor[ele]:
                    if nex not in horizon:
                        if inHorizon(iele, nex, *args):
                            lisNew.append(nex)
                            horizon.add(nex)
            lis = lisNew
        return horizon


    def inHorizon(self, iele, ele, *args):
        """
            judge whether ele is in the horizon of iele
            horizon has center at iele
                and is a sphere with specific radius
            input:
                iele: int, the center element id of horizon
                ele: int, element id to be judged whether inside the horizon
            output:
                variable of True or False
        """
        if len(args) == 0:
            radius = 7.5  # 7.5 is better for stress fitting of whole body
        else:
            radius = args[0]
        eps = 0.01  # relative error of eLen
        lenRange = (radius + eps) * self.eLen
        if sum((tch.tensor(self.eleCenter(ele)) - tch.tensor(self.eleCenter(iele)))**2) < lenRange**2:
            return True
        else:
            return False
    
    def inBox(self, iele, ele, *args):
        """
            judge whether ele is in the horizon of iele
            horizon has center at iele
                and has length of 11 * eLen at all 3 directions
            input:
                iele: int, the center element id of horizon
                ele: int, element id to be judged whether inside the horizon
            output:
                variable of True or False
            note:
                use self.eCen instead of self.eleCenter(),
                because each time call eleCenter() consumes an 'if' operation
                which slows down the process if we need centers frequently
        """
        if len(args) == 0:
            halfLength = 5.5
        else:
            halfLength = args[0]
        eCen = self.eCen
        eps = 0.01  # relative error of eLen
        lenRange = (halfLength + eps) * self.eLen
        if abs(eCen[ele][0] - eCen[iele][0]) <= lenRange:
            if abs(eCen[ele][1] - eCen[iele][1]) <= lenRange:
                if abs(eCen[ele][2] - eCen[iele][2]) <= lenRange:
                    return True
        return False


    def inSphericalHorizon(self, iele, ele, *args):
        """
            judge whether ele is in the horizon of iele
            horizon has center at iele
            input:
                iele: int, the center element id of horizon
                ele: int, element id to be judged whether inside the horizon
                *args: the radius of the horizon
            output:
                variable of True or False
            note:
                use self.eCen instead of self.eleCenter(),
                because each time call eleCenter() consumes an 'if' operation
                which slows down the process if we need centers frequently
        """
        ### radius: relative length
        if len(args) == 0:
            radius = 5.5
        else:
            radius = args[0]
        eCen = self.eCen
        eps = 0.01  # relative error of eLen
        lenRange = (radius + eps) * self.eLen
        cen1 = tch.tensor(eCen[ele])
        cen2 = tch.tensor(eCen[iele])
        if sum((cen1 - cen2)**2) <= lenRange ** 2:
            return True
        return False
    

    def ratioOfVisualize(self):
        """
            get the retio of visualization for this body
        """
        # get the maximum x and minimum 
        beg = time.time()
        xMax = max(self.nodes.values(), key=lambda x: x[0])[0]
        xMin = min(self.nodes.values(), key=lambda x: x[0])[0]
        yMax = max(self.nodes.values(), key=lambda x: x[1])[1]
        yMin = min(self.nodes.values(), key=lambda x: x[1])[1]
        zMax = max(self.nodes.values(), key=lambda x: x[2])[2]
        zMin = min(self.nodes.values(), key=lambda x: x[2])[2]
        print(
            "\033[40;35m{} \033[40;33m{}, \033[40;35m{} \033[40;33m{}\n"
            "\033[40;35m{} \033[40;33m{}, \033[40;35m{} \033[40;33m{}\n"
            "\033[40;35m{} \033[40;33m{}, \033[40;35m{} \033[40;33m{}\033[0m".format(
                "xMax =", xMax, "xMin =", xMin,
                "yMax =", yMax, "yMin =", yMin,
                "zMax =", zMax, "zMin =", zMin,
            )
        )
        print("time for max min computing =", time.time() - beg)
        self.maxLen = max(xMax - xMin, yMax - yMin, zMax - zMin)
        
        self.ratio_draw = 1. if self.maxLen < 10. else 5. / self.maxLen
        print("\033[35;1m{} \033[40;33;1m{}\033[0m".format(
            "this body's ratio of visualization (self.ratio_draw) =", self.ratio_draw
        ))
        self.regionCen = [(xMin + xMax) / 2., (yMin + yMax) / 2., (zMin + zMax) / 2.]
        return self.ratio_draw, self.regionCen


    def ele_directional_range(self, iele, direction=[0., 0., 1.]):
        """
            get the elements min or max coordiantes along a direction
            aim for computation of twin length
            return:
                minVal, maxVal
            author: mo-hanxuan@sjtu.edu.cn
            LICENSE: Apache license
        """
        ### unitize the direction vector
        eps = 1.e-6  # a value nearly 0
        drc = tch.tensor(direction)
        drc_len = (drc**2).sum() ** 0.5  # vector length of direction
        if drc_len < eps:
            raise ValueError(
                "input direction vector is almost 0, "
                "can not comput its length neumerically"
            )
        drc = drc / drc_len  # unitize

        def drcMinMax(id, drc):
            ### compute directional min and max value of an element,
            ### where input dic must be unitized
            nodesCoo = tch.tensor(
                [self.nodes[int(i)] for i in self.elements[id]]
            )
            minVal = min(nodesCoo, key=lambda xyz: xyz @ drc) @ drc
            maxVal = max(nodesCoo, key=lambda xyz: xyz @ drc) @ drc
            return float(minVal), float(maxVal)

        if isinstance(iele, int):
            return drcMinMax(iele, drc)
        elif iele == "all":
            minVals, maxVals = [], []
            for i in range(len(self.elements)):
                if i % 100 == 0:
                    progressBar_percentage(i / len(self.elements) * 100.)
                min_, max_ = drcMinMax(i, drc)
                minVals.append(min_)
                maxVals.append(max_)
            print("")  # break line for progress bar
            return minVals, maxVals
        else:
            raise ValueError("iele must be an int, or string 'all' to include all elements")


if __name__ == '__main__':

    # fileName = 'input/tilt45.inp'
    fileName = input("\033[40;33;1m{}\033[0m".format(
        "please input the .inp file name (no prefix): "
    ))

    body1 = ElementsBody(
        *readInp(fileName="input/" + fileName + ".inp")
    )
    print('elemens.size() =', body1.elements.size())
    print('len(nodes) =', len(body1.nodes))

    # ==============================================================
    # calssify the element facet into x-type, y-type, and z-type
    # ==============================================================
    print('\n --- now get all facets with their type ---')

    tBeg = time.time()
    body1.get_allFacets()  # get the facets information
    tEnd = time.time()
    print('\n facets done, consuming time is {} \n ---'.format(tEnd - tBeg))

