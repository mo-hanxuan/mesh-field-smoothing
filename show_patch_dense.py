"""
nonlinear fitting

before fitting, the sample points are densified,
then the dense samples are taken to fitting
"""

from attr import has
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Process

from facetDenseRegular import *
from neuralNetwork import NeuralNetwork

import sys

from torch._C import InterfaceType
from decideVfByGeometry import decideVfByGeometry
from decide_ratio_draw import decide_ratio_draw
from elementsBody import *

sys.path.append('./tryOpenGL')

import mhxOpenGL
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import coordinateLine
import colorBar
import Arrow3D, lightSource
from sphere3D import putSphere

"""different nonlinear function for fitting"""
# import circleFit as nonlinear
# import neuralNetwork as nonlinear
# import ellipticFit as nonlinear
# import logisticCum as nonlinear

from weightedAverage import *
from horizon import *


class bodyPatch(object):

    def __init__(self, obj1) -> None:
        if not isinstance(obj1, ElementsBody):
            raise ValidationErr("error, not isinstance(obj1, ElementsBody) ")
        self.obj1 = obj1
        self.region_cen = obj1.region_cen


    def get_ele_VfNet(self, iele, densify=True):
        """ 
            get the neural network for fitting of VF at element iele, 
            get the neural networks of the element
        """
        obj = self.obj1
        if not hasattr(obj, "eleNeighbor"):
            obj.get_eleNeighbor()
        if not hasattr(obj, "eLen"):
            obj.get_eLen()

        net = NeuralNetwork(mod="phi", dm=2)
        horizon = obj.findHorizon(iele=iele, inHorizon=obj.inBox)
        
        """ densify the sample for fitting """
        denseSamples = {}  # keys: element number
                           # values: dense nodes' coordinates attatched to this element
        if densify:
            denseSamples = get_denseSamples(obj, horizon)
        
            """ get dense Samples' values """
            spreadRange = float(input("\033[35;1m {} \033[0m".format("spread range = ")))
            denseSamplesVals = {}
            for iele in denseSamples:
                denseSamplesVals[iele] = []
                for nodeCoo in denseSamples[iele]:
                    denseSamplesVals[iele].append(
                        weighted_average(
                            xyz=nodeCoo, iele=iele,
                            obj1=obj, 
                            func="normal_distri", 
                            spreadRange=spreadRange,
                        )
                    )
            obj.denseSamples, obj.denseSamplesVals = denseSamples, denseSamplesVals
        
        """ fit the nets """
        xyzs, vals = [], []
        for other in horizon:
            if other in denseSamples:
                ### take the dense Samples
                for nodeCoo in denseSamples[other]:
                    xyzs.append(nodeCoo)
                for nodeVal in denseSamplesVals[other]:
                    vals.append(nodeVal)
            else:
                ### take center node with value of weighted average
                xyzs.append(np.array(obj.eleCenter(other)))
                vals.append(obj.VF[other])  # vals.append(averageVF(obj, other, spreadRange))
        net.getFitting(  # anagolus to the learning process by gradient decent method
            xyzs, vals, 
            region_cen=obj.eleCenter(iele),
            loss_tolerance=0.4,
            innerLoops=1,  # whether fit multiple times in the inner loop and choose a net with lowest loss
            plotData=True, printData=True,  # plotData=False, printData=False, 
            frozenClassifier=False,  
                ### frozenClassifier, whether freeze the classier (possiotion and direction), 
                ### so that the wieghts and bias at hidden layer remian unchanged 
            prematurelyBreak=False,  # whether prematurely break when the parameters nearly unchanged at a step
            refitTimes=10,  # refit by changing the initial values of weights
        )
        return net
    

    def get_ele_stressNet(self, iele, vfNet=None, densify=False, frozenClassifier=False):
        """ 
            get the neural network for fitting of stress at element iele, 
            return the neural networks of the element
        """
        obj = self.obj1
        if not hasattr(obj, "eleNeighbor"):
            obj.get_eleNeighbor()
        if not hasattr(obj, "eLen"):
            obj.get_eLen()

        net = NeuralNetwork(mod="stress", dm=2)
        horizon = obj.findHorizon(iele=iele, inHorizon=obj.inBox)
        
        """ denseSamples, keys: element number
                          values: dense nodes' coordinates attatched to this element"""
        denseSamples = {}

        if densify:
            denseSamples = get_denseSamples(obj, horizon)
            """ get dense Samples' values """
            spreadRange = float(input("\033[35;1m {} \033[0m".format("spread range = ")))
            denseSamplesVals = {}
            for iele in denseSamples:
                denseSamplesVals[iele] = []
                for nodeCoo in denseSamples[iele]:
                    denseSamplesVals[iele].append(
                        weighted_average(
                            xyz=nodeCoo, iele=iele,
                            obj1=obj, 
                            func="normal_distri", 
                            spreadRange=spreadRange,
                            field=obj.stress
                        )
                    )
        
        """ fit the nets """
        xyzs, vals = [], []
        for other in horizon:
            if other in denseSamples:
                ### take the dense Samples
                for nodeCoo in denseSamples[other]:
                    xyzs.append(nodeCoo)
                for nodeVal in denseSamplesVals[other]:
                    vals.append(nodeVal)
            else:
                ### take center node with value of weighted average
                xyzs.append(np.array(obj.eleCenter(other)))
                vals.append(obj.stress[other])  # vals.append(averageVF(obj, other, spreadRange))
        
        if vfNet != None:  # initial weights of stress' neural network
            initialWei = list(map(lambda x: x.data, vfNet.net.parameters()))
            initialWei[-2] = -initialWei[-2]
            initialWei[-1][:] = sum(vals) / len(vals)
        else:
            initialWei = None

        net.getFitting(  # anagolus to the learning process by gradient decent method
            xyzs, vals, 
            lr=0.2,  # learning rate
            region_cen=obj.eleCenter(iele),
            loss_tolerance=((max(vals) - min(vals)) / 3.)**2,
            initialWei=initialWei,
            loopMax=2500,
            innerLoops=1,  # whether fit multiple times in the inner loop and choose a net with lowest loss
            plotData=True, printData=True,  # plotData=False, printData=False, 
            frozenClassifier=frozenClassifier,  
                ### frozenClassifier, whether freeze the classier (possiotion and direction), 
                ### so that the wieghts and bias at hidden layer remian unchanged 
            prematurelyBreak=False,  # whether prematurely break when the parameters nearly unchanged at a step
            refitTimes=10,  # refit by changing the initial values of weights
        )
        return net


    def simple_draw_patch_byFit_VF(self, iele, densify=True, drawArrows=False):
        obj = self.obj1
        net = self.get_ele_VfNet(iele, densify)
        self.simple_draw_patch(obj.VF, 
                               minVal=None, maxVal=None,
                               iele=iele,
                               valueFunc=net.reasoning,
                               drawArrows=drawArrows,
                               gradFunc=net.getGradient)
    

    def simple_draw_patch_byFit_stress(self, iele, 
                                       minVal=None, maxVal=None, 
                                       densify=False, drawArrows=False, 
                                       frozenClassifier=False):
        obj = self.obj1
        vfNet = self.get_ele_VfNet(iele, densify=True)
        stressNet = self.get_ele_stressNet(iele, vfNet=vfNet, densify=densify, 
                                           frozenClassifier=frozenClassifier)
        self.simple_draw_patch(obj.stress, 
                               minVal=minVal, maxVal=maxVal,
                               iele=iele,
                               valueFunc=stressNet.reasoning,
                               drawArrows=drawArrows,
                               gradFunc=stressNet.getGradient)
    

    def simple_draw_patch(self, field, 
                          minVal=None, maxVal=None,
                          iele=0,
                          valueFunc=weighted_average,
                          drawArrows=False,
                          gradFunc=lambda x: 0):
        """
            draw the patch at the region center at iele
        """
        obj = self.obj1
        obj.get_outerFacetDic()
        obj.get_facialEdgeDic()
        
        minVal = min(field) if minVal == None else minVal
        maxVal = max(field) if maxVal == None else maxVal

        patchEles = horizons(obj, iele, inHorizon=obj.inBox)
        facets = {}
        for facet in obj.outerFacetDic:
            ele = obj.outerFacetDic[facet]
            if ele in patchEles:
                facets[facet] = {}
                ### densify or not
                sameNeighbor = True
                for other in obj.eleNeighbor[ele]:
                    if obj.VF[other] != obj.VF[ele]:
                        sameNeighbor = False
                        break
                if not sameNeighbor:
                    denseOrder = 4  # densify by 4 times
                else:
                    denseOrder = 1
                ### get the densified facets inside big facets
                facets[facet]["denseNodesCoo"], facets[facet]["facets"], _ = facetDenseRegular(
                    np.array([obj.nodes[node] for node in facet]), order=denseOrder
                )
                facets[facet]["fieldVals"] = {}
                for node in facets[facet]["denseNodesCoo"]:
                    facets[facet]["fieldVals"][node] = float(valueFunc(facets[facet]["denseNodesCoo"][node]))
        edges = set()
        for edge in obj.facialEdgeDic:
            for facet in obj.facialEdgeDic[edge]:
                if facet in facets:
                    edges.add(edge)
                    break
        
        ### whether draw the arrows of gradients
        gradients = {}  # key: position, value: gradient
        if drawArrows:
            eps = 0.01
            for ele in patchEles:
                if obj.VF[ele] < 1.-eps:
                    neighborOne = False
                    for other in obj.eleNeighbor[ele]:
                        if obj.VF[other] > 1.-eps:
                            neighborOne = True
                            break
                    if neighborOne:  # self small, neighbor = 1
                        gradients[tuple(obj.eleCenter(ele))] = gradFunc(obj.eleCenter(ele))
        Process(
            target=mhxOpenGL.showUp, 
            args=(self.__simple_draw_patch__, 
                  facets, edges, 
                  minVal, maxVal,
                  np.array(obj.eleCenter(iele)),  # regionCen
                  gradients)
        ).start()
    

    def __simple_draw_patch__(self, facets, edges, 
                              minVal, maxVal, 
                              regionCen,
                              gradients):
        obj = self.obj1
        ### ----------------------------------- draw the facets
        glBegin(GL_QUADS)
        for bigFacet in facets:
            for smallFacet in facets[bigFacet]["facets"]:
                for node in smallFacet:
                    VF = facets[bigFacet]["fieldVals"][node]
                    color = (VF - minVal) / (maxVal - minVal)
                    red, green, blue = colorBar.getColor(color)
                    glColor4f(red, green, blue, 1.0)
                    glMaterialfv(GL_FRONT, GL_AMBIENT, [red, green, blue])
                    glMaterialfv(GL_FRONT, GL_DIFFUSE, [red, green, blue])
                    glMaterialfv(GL_FRONT, GL_SPECULAR, [red, green, blue])
                    glMaterialfv(GL_FRONT, GL_EMISSION, [red, green, blue])
                    glVertex3f(
                        (facets[bigFacet]["denseNodesCoo"][node][0] - regionCen[0]) * obj.ratio_draw, 
                        (facets[bigFacet]["denseNodesCoo"][node][1] - regionCen[1]) * obj.ratio_draw, 
                        (facets[bigFacet]["denseNodesCoo"][node][2] - regionCen[2]) * obj.ratio_draw
                    )
        glEnd()

        ### ----------------------------------- draw the element edges
        glLineWidth(2.)
        red, green, blue = 0.4, 0.4, 0.4
        glColor4f(red, green, blue, 1.0)
        glMaterialfv(GL_FRONT, GL_AMBIENT, [red, green, blue])
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [red, green, blue])
        glMaterialfv(GL_FRONT, GL_SPECULAR, [red, green, blue])
        glMaterialfv(GL_FRONT, GL_EMISSION, [red, green, blue])
        glBegin(GL_LINES)
        for edge in edges:
            glVertex3f((obj.nodes[edge[0]][0] - regionCen[0]) * obj.ratio_draw, 
                       (obj.nodes[edge[0]][1] - regionCen[1]) * obj.ratio_draw, 
                       (obj.nodes[edge[0]][2] - regionCen[2]) * obj.ratio_draw)
            glVertex3f((obj.nodes[edge[1]][0] - regionCen[0]) * obj.ratio_draw, 
                       (obj.nodes[edge[1]][1] - regionCen[1]) * obj.ratio_draw, 
                       (obj.nodes[edge[1]][2] - regionCen[2]) * obj.ratio_draw)
        glEnd()

        ### ----------------------------------- draw the gradient arrows
        red, green, blue = 1., 1., 1.
        glColor4f(red, green, blue, 1.0)
        glMaterialfv(GL_FRONT, GL_AMBIENT, [red, green, blue])
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [red, green, blue])
        glMaterialfv(GL_FRONT, GL_SPECULAR, [0., 1., 1.])
        glMaterialfv(GL_FRONT, GL_EMISSION, [0., 0., 0.])
        for position in gradients:
            r = np.array([*position[:2], 1.1]) - regionCen
            r *= obj.ratio_draw
            r = r.tolist()
            h = 1.5 * obj.ratio_draw  # use same length to visualize arrows

            grad = gradients[position]
            angle = np.degrees(np.arctan(grad[1] / grad[0]))
            if np.sign(grad[0]) == 0:
                angle = np.sign(grad[1]) * 90.
            if grad[0] < 0:
                angle -= 180.
            angle -= 180.
            Arrow3D.show(h, 0.05, angle, r)
    
        ### ---------------------------------------------------------------
        glutSwapBuffers()  # 切换缓冲区，以显示绘制内容
 
    
    def simple_draw_denseNodes_of_patch(self, iele, densify=True, minVal=None, maxVal=None):
        obj = self.obj1
        nodes = {}
        if densify:
            if not hasattr(obj, "denseSamples"):
                obj.denseSamples = get_denseSamples(
                    obj=obj, 
                    horizon=horizons(obj, iele, inHorizon=obj.inBox)
                )
                """ get dense Samples' values """
                spreadRange = float(input("\033[35;1m {} \033[0m".format("spread range = ")))
                denseSamplesVals = {}
                for ele in obj.denseSamples:
                    denseSamplesVals[ele] = []
                    for nodeCoo in obj.denseSamples[ele]:
                        denseSamplesVals[ele].append(
                            weighted_average(
                                xyz=nodeCoo, iele=ele,
                                obj1=obj, 
                                func="normal_distri", 
                                spreadRange=spreadRange,
                            )
                        )
                obj.denseSamplesVals = denseSamplesVals
            for ele in obj.denseSamples:
                for node in range(len(obj.denseSamples[ele])):
                    nodes[tuple(obj.denseSamples[ele][node])] = obj.denseSamplesVals[ele][node]
            for ele in horizons(obj, iele=iele, inHorizon=obj.inBox):
                if ele not in obj.denseSamples:
                    nodes[tuple(obj.eleCenter(ele))] = obj.VF[ele]
        else:
            for ele in horizons(obj, iele=iele, inHorizon=obj.inBox):
                nodes[tuple(obj.eleCenter(ele))] = obj.VF[ele]
        
        minVal = min(obj.VF) if minVal == None else minVal
        maxVal = max(obj.VF) if maxVal == None else maxVal
        
        Process(
            target=mhxOpenGL.showUp,
            args=(
                self.__draw_nodes_of_patch__,
                nodes, minVal, maxVal,
                obj.eleCenter(iele),  # regionCen
            )
        ).start()


    def __draw_nodes_of_patch__(self, nodes, minVal, maxVal, regionCen):
        """
        input: nodes -> Dict
            key: position
            value: field value of this node
        """
        obj = self.obj1
        for position in nodes:
            red, green, blue = colorBar.getColor((nodes[position] - minVal) / (maxVal - minVal))
            glColor4f(red, green, blue, 1.0)
            glMaterialfv(GL_FRONT, GL_AMBIENT, [red, green, blue])
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [red, green, blue])
            glMaterialfv(GL_FRONT, GL_SPECULAR, [red, green, blue])
            glMaterialfv(GL_FRONT, GL_EMISSION, [red, green, blue])

            radius = 0.11
            pos = np.array(position) - np.array([*regionCen[:2], -0.5])
            pos, radius = pos * obj.ratio_draw, radius * obj.ratio_draw
            putSphere(pos, radius, [red, green, blue], resolution=10)
        ### ---------------------------------------------------------------
        glutSwapBuffers()  # 切换缓冲区，以显示绘制内容


    def simple_draw_patch_byOriginal(self, iele, field, minVal=None, maxVal=None, drawArrows=False):
        """
            draw the patch by original field value
        """
        obj = self.obj1
        obj.get_outerFacetDic()
        obj.get_facialEdgeDic()
        
        minVal = min(field) if minVal == None else minVal
        maxVal = max(field) if maxVal == None else maxVal

        facets, edges = horizonsFacets(obj, iele, inHorizon=obj.inBox)
        for facet in facets:
            facets[facet]["denseNodesCoo"], facets[facet]["facets"], _ = facetDenseRegular(
                np.array([obj.nodes[node] for node in facet]), order=1
            )
            facets[facet]["fieldVals"] = {}
            for node in facets[facet]["denseNodesCoo"]:
                facets[facet]["fieldVals"][node] = field[obj.outerFacetDic[facet]]

        gradients = {}
        Process(
            target=mhxOpenGL.showUp, 
            args=(self.__simple_draw_patch__, 
                  facets, edges, 
                  minVal, maxVal,
                  np.array(obj.eleCenter(iele)),  # regionCen
                  gradients)
        ).start()


def get_denseSamples(obj, horizon):
    if not isinstance(obj, ElementsBody):
        raise ValueError("error, not isinstance(obj, ElementsBody) ")
    denseSamples = {}  # keys: element number
                       # values: dense nodes' coordinates attatched to this element
    vx, vy = np.array([1., 0., 0.]) * obj.eLen, np.array([0., 1., 0.]) * obj.eLen
    for iele in horizon:
        sameNeighbor = True
        for other in obj.eleNeighbor[iele]:
            if obj.VF[other] != obj.VF[iele]:
                sameNeighbor = False
                break
        if not sameNeighbor:
            denseSamples[iele] = []
            ### get the dense nodes' coordinates for this element
            for i in [-1./3., 0., 1./3.]:
                for j in [-1./3., 0., 1./3.]:
                    denseSamples[iele].append(
                        np.array(obj.eleCenter(iele)) + i * vx + j * vy
                    )
    return denseSamples


def get_denseSamplesVals(obj, denseSamples):
    """ get dense Samples' values """
    spreadRange = float(input("\033[35;1m {} \033[0m".format("spread range = ")))
    denseSamplesVals = {}
    for iele in denseSamples:
        denseSamplesVals[iele] = []
        for nodeCoo in denseSamples[iele]:
            denseSamplesVals[iele].append(
                weighted_average(
                    xyz=nodeCoo, iele=iele,
                    obj1=obj, 
                    func="normal_distri", 
                    spreadRange=spreadRange,
                    field=obj.stress
                )
            )
    return denseSamplesVals


if __name__ == '__main__':
    w = [6.2560, -2.5243,  7.5602,  2.0000]
    x = [10., 20.]

    inpFile = input("\033[40;35;1m{}\033[0m".format(
        "please give the .inp file name (include the path): "
    ))
    obj1 = ElementsBody(
        *readInp(fileName=inpFile)
    )
    print("obj1.elements.size() = \033[40;33;1m{}\033[0m".format(obj1.elements.size()))

    celent = obj1.get_eLen()  # characteristic element length

    # decide the ratio_draw of this object
    decide_ratio_draw(obj1)

    """center coordinates of the selected region"""
    # region_cen = np.array([0.5, 0.5])
    # region_cen = np.array([10., 2.5 * 3.**0.5])
    region_cen = np.array([20.5, 0.5])

    # ### ================================= set the volume fraction for the elements
    # decideVfByGeometry(obj1, mod="constrainedSharp", geometry="ellip")
    
    ### ================================= set the volume fraction and other field values for the elements
    dataFile = input("\033[40;35;1m{}\033[0m".format(
        "please give the data file name (include the path): "))
    dataFrame = readDataFrame(fileName=dataFile)
    frame = int(input("which frame do you want? frame = "))
    obj1.VF = dataFrame["SDV210_frame{}".format(frame)]
    obj1.stress = dataFrame["SDV212_frame{}".format(frame)]
    
    obj1.region_cen = region_cen
    body1 = bodyPatch(obj1)  # ignite ---------------------------------------------------------
    
    ### show the patch by fit values
    iele = min(
        range(len(obj1.elements)),
        key=lambda x:
            sum((np.array(obj1.eleCenter(x)[:2]) - region_cen)**2)
    )
    body1.simple_draw_patch_byOriginal(iele, field=obj1.VF)
    body1.simple_draw_patch_byOriginal(
        iele, field=obj1.stress, 
        minVal=-550., 
        # maxVal=700.,
    )
    body1.simple_draw_patch_byFit_VF(iele, densify=False, drawArrows=True)
    body1.simple_draw_patch_byFit_VF(iele, densify=True, drawArrows=True)
    body1.simple_draw_denseNodes_of_patch(iele)
    body1.simple_draw_patch_byFit_stress(iele, minVal=-550., maxVal=700.)
    