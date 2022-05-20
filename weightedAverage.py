from xml.dom import ValidationErr
from numpy.lib.function_base import cov
import torch as tch
import numpy as np

from elementsBody import ElementsBody


def weighted_average(
        xyz, iele, obj1, 
        func="1/(1+x^2)",
        result='val',  # return value or return grad
        dimension=2,  # dimension for normal distribution
        spreadRange=1.5,
        field=None,
    ):
    """
        goal: get the field value (or gradient) of weighted average field of a given node

        input:
            xyz coordinates of the node, 
            iele (element number) that the node comes from
            obj1: the object contains many elements (of class ElementsBody())
    """
    if not isinstance(obj1, ElementsBody):
        raise ValidationErr("error, not isinstance(obj1, ElementsBody) ")
    if not hasattr(obj1, "eleNeighbor"):
        obj1.get_eleNeighbor()
    if iele < 0 or iele >= len(obj1.elements):
        raise ValidationErr("\033[31;1m {} {} \033[0m".format(
            "error, index iele out of range, iele =", iele
        ))
    if type(field) not in [type(np.array([])), type(tch.tensor([])), type([])]:
        field = obj1.VF
    weightFuncs = {}
    dwdxs = {}

    ### ---------------------------------------------------- things for func1
    def func1(dis):  # the weighted function of 1/(1+x^2)
        return 1. / (1. + (dis**2).sum())
    def dwdx_func1(dis): 
        """d(func1) / d(xyz)"""
        return 2. * dis / (1. + (dis**2).sum())**2
    weightFuncs["1/(1+x^2)"] = func1
    dwdxs["1/(1+x^2)"] = dwdx_func1

    ### ---------------------------------------------------- things for func2
    eLen = obj1.getVolumes(iele)**(1./3.)
    std_deviate = spreadRange * eLen / 3.  # standard deviate σ, f(3σ) ≈ 0
    sigmaX = 5.5 * eLen / 3.
    sigmaY = 1.5 * eLen / 3.
    covariance = tch.tensor([  # covariance matrix
        [std_deviate**2, 0., 0.],
        [0., std_deviate**2, 0.],
        [0., 0., std_deviate**2],
    ], dtype=tch.float64)
    def func2(dis):  # normal distribution
        dis = tch.tensor(dis, dtype=tch.float64)
        preFact = (
            (2. * np.pi)**dimension 
            * tch.det(covariance[:dimension, :dimension])
        ) ** (-1./2.)
        if dimension == 2:
            return preFact * float(tch.exp(
                -0.5 * dis[:2] @ covariance[:2, :2].pinverse() @ dis[:2],
            ))
        else:
            return preFact * float(tch.exp(
                -0.5 * dis @ covariance.pinverse() @ dis,
            ))
    def dwdx_func2(dis):
        """d(func2) / d(xyz)"""
        dis = tch.tensor(dis, dtype=tch.float64)
        if dimension == 2:
            return func2(dis) * covariance[:2, :2].pinverse() @ dis[:2]
        else:
            return func2(dis) * covariance.pinverse() @ dis
    weightFuncs["normal_distri"] = func2
    dwdxs["normal_distri"] = dwdx_func2
    ### ---------------------------------------------------- end of things for funcs

    while result not in ['val', 'grad']:
        result = input("\033[40;31;1m{}\033[0m".format(
            "please choose a result type (either 'val' or 'grad'): "
        ))
    if result == 'val':
        weights, eVals = [], []
        
        if spreadRange < 1.6:
            eles = obj1.eleNeighbor[iele] | {iele}
        else:
            eles = obj1.findHorizon(iele, obj1.inSphericalHorizon, spreadRange)
        if len(eles) == 0:
            raise ValidationErr("len(eles) = 0, no horizon is found, iele = {} ".format(iele))

        for ele in eles:
            dis = np.array(obj1.eleCenter(ele)) - np.array(xyz)
            weights.append(
                (
                    weightFuncs[func](dis)
                ) * obj1.getVolumes(eleNum=ele)
            )
            eVals.append(field[ele])
        weights = np.array(weights)
        eVals = np.array(eVals)
        return sum(weights * eVals) / weights.sum()
    elif result == 'grad':
        weights, eVals = [], []
        dwei_dx = []
        if spreadRange < 1.6:
            eles = obj1.eleNeighbor[iele] | {iele}
        else:
            eles = obj1.findHorizon(iele, obj1.inSphericalHorizon, spreadRange)

        for ele in eles:
            dis = np.array(obj1.eleCenter(ele)) - np.array(xyz)
            weights.append(
                (
                    weightFuncs[func](dis)
                ) * obj1.getVolumes(eleNum=ele)
            )
            eVals.append(field[ele])
            dwei_dx.append(
                (
                    dwdxs[func](dis) * obj1.getVolumes(eleNum=ele)
                ).tolist()
            )
        weights = tch.tensor(weights, dtype=tch.float64)
        eVals = tch.tensor(eVals, dtype=tch.float64)
        dwei_dx = tch.tensor(dwei_dx, dtype=tch.float64)
        A = tch.dot(weights, eVals)
        B = weights.sum()
        dAdx = tch.einsum("ij, i -> j", dwei_dx, eVals)
        dBdx = dwei_dx.sum()
        return (B * dAdx - A * dBdx) / B**2


def weighted_average_forStress(
        xyz, iele, obj1, 
        func="1/(1+x^2)",
        result='val',  # return value or return grad
        dimension=2,  # dimension for normal distribution
        nets=None,  # the pretrained neural network of phi
        field=None,
        spreadRange=1.5,
    ):
    """
        in weighted average for stress,
        we should specify some neighbors
        which satisfy same grain and same phi with the given node,

        particularly, during loading (or relax) process, 
            if phi < 1 in dense nodes, 
            we consider neighbors with phi ≈ 0
        unloading process obey the opposite rule

        input:
            xyz coordinates of the node, 
            iele (element number) that the node comes from
            obj1: the object contains many elements (of class ElementsBody())
    """
    weightFuncs = {}
    dwdxs = {}

    if field == None:
        field = obj1.stress

    ### ---------------------------------------------------- things for func1
    def func1(dis):  # the weighted function of 1/(1+x^2)
        return 1. / (1. + (dis**2).sum())
    def dwdx_func1(dis): 
        """d(func1) / d(xyz)"""
        return 2. * dis / (1. + (dis**2).sum())**2
    weightFuncs["1/(1+x^2)"] = func1
    dwdxs["1/(1+x^2)"] = dwdx_func1

    ### ---------------------------------------------------- things for func2
    std_deviate = 0.5  # standard deviate σ, f(3σ) ≈ 0
    covariance = tch.tensor([  # covariance matrix
        [std_deviate**2, 0., 0.],
        [0., std_deviate**2, 0.],
        [0., 0., std_deviate**2],
    ], dtype=tch.float64)
    def func2(dis):  # normal distribution
        dis = tch.tensor(dis, dtype=tch.float64)
        preFact = (
            (2. * np.pi)**dimension 
            * tch.det(covariance[:dimension, :dimension])
        ) ** (-1./2.)
        if dimension == 2:
            return preFact * float(tch.exp(
                -0.5 * dis[:2] @ covariance[:2, :2].pinverse() @ dis[:2],
            ))
        else:
            return preFact * float(tch.exp(
                -0.5 * dis @ covariance.pinverse() @ dis,
            ))
    def dwdx_func2(dis):
        """d(func2) / d(xyz)"""
        dis = tch.tensor(dis, dtype=tch.float64)
        if dimension == 2:
            return func2(dis) * covariance[:2, :2].pinverse() @ dis[:2]
        else:
            return func2(dis) * covariance.pinverse() @ dis
    weightFuncs["normal_distri"] = func2
    dwdxs["normal_distri"] = dwdx_func2
    ### ---------------------------------------------------- end of things for funcs

    while result not in ['val', 'grad']:
        result = input("\033[40;31;1m{}\033[0m".format(
            "please choose a result type (either 'val' or 'grad'): "
        ))
    if result == 'val':
        weights, eVals = [], []
        if spreadRange < 1.6:
            eles = list(obj1.eleNeighbor[iele] | {iele})
        else:
            eles = obj1.findHorizon(iele, obj1.inSphericalHorizon, spreadRange)
        
        ### smoothing only takes stresses of elements with same VF
        # idx = 0
        # while idx < len(eles):
        #     if abs(obj1.VF[eles[idx]] - obj1.VF[iele]) > 1.e-3:  # if obj1.VF[eles[idx]] != obj1.VF[iele]:
        #         del eles[idx]
        #     else:
        #         idx += 1

        # ### another way: smoothing only takes stresses of elements with same character (≈ 0, ≈ 1, or 0 < VF < 1)
        # eps = 1.e-3
        # elesNew = []
        # if obj1.VF[iele] < eps:
        #     for ele in eles:
        #         if obj1.VF[ele] < eps:
        #             elesNew.append(ele)
        # elif obj1.VF[iele] > 1.-eps:
        #     for ele in eles:
        #         if obj1.VF[ele] > 1.-eps:
        #             elesNew.append(ele)
        # else:
        #     for ele in eles:
        #         if eps <= obj1.VF[ele] <= 1.-eps:
        #             elesNew.append(ele)
        # eles = elesNew

        # eles = sameGrainNeighbor_forStress(  # get the neighbors
        #     obj1, iele, 
        #     # VF=net.reasoning(xyz),
        #     VF=obj1.VF[iele], 
        #     mod="loading",
        #     spreadRange=spreadRange,
        # )
        for ele in eles:
            eleCenter = obj1.eCen[ele] if ele in obj1.eCen else obj1.eleCenter(ele)
            dis = np.array(eleCenter) - np.array(xyz)
            weights.append(
                (
                    weightFuncs[func](dis)
                ) * obj1.getVolumes(eleNum=ele)
            )
            eVals.append(field[ele])
        if len(weights) > 0:
            weights = np.array(weights)
            eVals = np.array(eVals)
            # return (weights * eVals).sum() / weights.sum()
            return (weights * eVals).sum()
        else:
            print("\033[31;1m{}\033[0m".format(
                "no eles meet requirements in neighbor region, return stress 0 as weightedAverage value"))
            return 0.  # no eles meet requirements in neighbor region, return stress 0 as weightedAverage value
    else:
        print("\033[31;1m{}\033[0m".format("gradient for stress has not been developed yet."))


def sameGrainNeighbor_forStress(
        obj1, iele, VF,
        mod="loading",
        spreadRange=1.5, 
    ):
    """
        in weighted average for stress,
        we should specify some neighbors
        which satisfy same grain and same phi,

        particularly, if phi < 1 in dense nodes, 
            we consider neighbors with phi ≈ 0 (during loading process)
    """
    eps = 1.e-3
    eLen = obj1.getVolumes(iele)**(1./3.)

    def inLayers(ele1, ele2):
        """
            judge whether ele2 is inside layers of ele1
        """
        cen1 = np.array(obj1.eleCenter(ele1))
        cen2 = np.array(obj1.eleCenter(ele2))
        if ((cen1 - cen2)**2).sum() < (spreadRange * eLen)**2:
            return True
        else:
            return False

    neis = []  # neighbors taken into account
    for nei in obj1.findHorizon(iele, inHorizon=inLayers):
        if chooseEle(obj1.VF[nei], mod):
            neis.append(nei)
    return neis 


def chooseEle(VF, mod="loading"):
    """
        loading or relax stage, 
            choose the elements at the outside of interface where phi ≈ 0
        unloading stage,
            choose the elements outside of parent where phi ≈ 1
    """
    eps = 1.e-3
    if mod in ["loading", "relax"]:
        if VF < eps:  # ≈ 0
            return True
        else:
            return False
    else:  # unloading
        if (1. - VF) < eps:   # ≈ 1
            return True
        else:
            return False