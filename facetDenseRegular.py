from xml.dom import ValidationErr
import numpy as np
import torch as tch


def shapeFunc2D(natCoo):
    """
            v1 ---- v2
            |        |
            |        |
     y ^    v0 ---- v3
       |
       ----> x
    """
    ntco0 = np.array([
        [-1., -1], [-1, 1], [1, 1], [1, -1]
    ])
    res = [0.25 for _ in range(4)]
    for node in range(4):
        for i in range(2):
            res[node] *= 1. + ntco0[node][i] * natCoo[i]
    if abs(sum(res) - 1.) > 1.e-8:
        print("res =", res, ", naturalCoo =", natCoo)
        print("ntco0 =", ntco0)
        raise ValueError("error, abs(sum(res) - 1.) > 1.e-8 ")
    return res


def facetDenseRegular(nodesCoo, order):
    """ input:
            4 nodes' coordiantes of the facet,
            densified order (how many times to densify)
        output:
            the nodes' coordinates of regular dense grid, 
            and the facet's corresponding nodes' number
    """

    if not isinstance(nodesCoo, np.ndarray) and not tch.is_tensor(nodesCoo):
        raise ValidationErr("error, not isinstance(nodesCoo, np.ndarray) and not tch.is_tensor(nodesCoo)")

    if len(nodesCoo) != 4:
        raise ValueError("error, len(nodesCoo) != 4 ")
    
    denseNodesCoo = {}
    step = 2. / order
    xs = np.arange(-1., 1.+step, step)
    for i in range(order + 1):
        for j in range(order + 1):
            denseNodesCoo[(i, j)] = np.array([xs[j], xs[i]])
            ### from natural coordinates to global coordinates
            denseNodesCoo[(i, j)] = np.einsum("n, nk -> k", shapeFunc2D(denseNodesCoo[(i, j)]), nodesCoo)
    
    facets = []
    for i in range(order):
        for j in range(order):
            facets.append([
                (i, j), (i+1, j),
                (i+1, j+1), (i, j+1)
            ])
    outerFrames = []
    for i in [0, order]:
        for j in range(1, order + 1):
            outerFrames.append([(i, j-1), (i, j)])
    for j in [0, order]:
        for i in range(1, order + 1):
            outerFrames.append([(i-1, j), (i, j)])
            
    return denseNodesCoo, facets, outerFrames


if __name__ == "__main__":

    natCoo = np.array([-0.5, -0.5])

    print("shapeFunc2D(natCcoo) =", shapeFunc2D(natCoo))


    order = 4
    nodes = np.zeros((order + 1, order + 1, 2), dtype=np.float64)
    step = 2. / order
    xs = np.arange(-1., 1.+step, step)
    for i in range(order + 1):
        for j in range(order + 1):
            nodes[i][j][1] = xs[i]
            nodes[i][j][0] = xs[j]
    
    print("nodes =\n", nodes)


    nodesCoo = np.array([
        [-1., -1], [-1, 1], [1, 1], [1, -1]
    ])
    nodes = {}
    step = 2. / order
    xs = np.arange(-1., 1.+step, step)
    for i in range(order + 1):
        for j in range(order + 1):
            nodes[(i, j)] = np.array([xs[j], xs[i]])
            ### from natural coordinates to global coordinates
            nodes[(i, j)] = np.einsum("n, nk -> k", shapeFunc2D(nodes[(i, j)]), nodesCoo)
    
    print("\033[35;1m nodes =\n", nodes)

    

