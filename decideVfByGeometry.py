from elementsBody import *
import numpy as np


def shapeFunc(naturalCoo):
    """ shape function of C3D8R element """
    ntco0 = np.array([
        [-1, -1, -1, -1,  1,  1,  1,  1],
        [ 1, -1, -1,  1,  1, -1, -1,  1],
        [ 1,  1, -1, -1,  1,  1, -1, -1]
    ]).transpose()

    res = np.array([1. / 8. for _ in range(8)])
    for node in range(len(res)):
        for j in range(3):
            res[node] *= 1. + ntco0[node][j] * naturalCoo[j]
    if abs(sum(res) - 1.) > 1.e-8:
        print("res =", res, ", naturalCoo =", naturalCoo)
        print("ntco0 =", ntco0)
        raise ValueError("error, abs(sum(res) - 1.) > 1.e-8 ")
    return res


def isTwin(xyz, geometry="ellip"):
    """
    input:
        xyz -> np.ndarray: coordinates of a point, 
        geometry -> string: geometric type of the twin
    output:
        bool: whether this point is inside the twin
    """
    if geometry == "ellip":
        longRadius, shortRadius = 20., 5.
        if (xyz[0] / longRadius)**2 + (xyz[1] / shortRadius)**2 < 1.**2:
            return True
        else:
            return False
    elif geometry == "flatEllip":
        longRadius, shortRadius = 40., 2.
        if (xyz[0] / longRadius)**2 + (xyz[1] / shortRadius)**2 < 1.**2:
            return True
        else:
            return False
    elif geometry == "smallCircle":
        longRadius, shortRadius = 2., 2.
        if (xyz[0] / longRadius)**2 + (xyz[1] / shortRadius)**2 < 1.**2:
            return True
        else:
            return False
    elif geometry == "line":
        slop = np.tan(-20. / 180. * np.pi)
        bias = 0.
        if (-slop * (xyz[0] - 0.5) + (xyz[1] - 0.5) + bias) < 0:
            return True
        else:
            return False


def decideVfByGeometry(obj, mod="constrainedSharp", geometry="ellip"):
    """
        get the volume fraction by 
        intersection of the inclusion and the element. 

        use descrete integration,
        integration points are evenly distributed in the element. 
    """
    if not isinstance(obj, ElementsBody):
        raise ValueError("error, not isinstance(obj, ElementsBody) ")
    denseSize = 8

    ### get the natural coordinates of insideNodes
    insideNodes = np.zeros((denseSize, denseSize, 2), dtype=np.array([1.]).dtype)
    for i in range(denseSize):
        for j in range(denseSize):
            insideNodes[i][j][0] = -1. + 1./denseSize + i * 2./denseSize
            insideNodes[i][j][1] = -1. + 1./denseSize + j * 2./denseSize

    VF = np.zeros((len(obj.elements)), dtype=np.array([1.]).dtype)

    if mod == "constrainedSharp":  # 0 < VF < 1 at the interface, VF is decided by integral
        for iele in range(len(obj.elements)):
            ### compute whether 8 nodes are twin or not
            nodesCoors = [np.array(obj.nodes[int(i)]) for i in obj.elements[iele]]
            nodeIsTwin = [isTwin(nodesCoo, geometry) for nodesCoo in nodesCoors]
            ### judge whether the element is fully inside/outside the ellip region
            if any(nodeIsTwin) == False:
                VF[iele] = 0.
            elif all(nodeIsTwin) == True:
                VF[iele] = 1.
            else:
                ### -------- compute the final volume fraction by integral (densify an element by n x n)
                ### get the global coordinates of insideNodes
                theVF = 0.
                for i in range(denseSize):
                    for j in range(denseSize):
                        globalCoor = np.einsum(
                            "i, ij -> j", 
                            shapeFunc(np.append(insideNodes[i][j], 0.)),
                            np.array(nodesCoors)
                        )
                        if isTwin(globalCoor, geometry):
                            theVF += 1
                VF[iele] = theVF / denseSize**2
    elif mod == "simplifiedSharp":  # either 0 or 1 for VF
        for iele in range(len(obj.elements)):
            if isTwin(obj.eleCenter(iele), geometry):
                VF[iele] = 1.
            else:
                VF[iele] = 0.
    else:
        raise ValueError("error, args mod can either be 'constrainedSharp' or 'simplifiedSharp'")

    obj.VF = VF