"""
    show entire body by locally densified, two options can be chosen:
        (1) local densified nodes
        (2) local densified field, with coarse edges to be visualized
"""

from re import S
from tokenize import Double
from xml.dom import ValidationErr
from attr import has
import numpy as np
import torch as th
from decide_ratio_draw import decide_ratio_draw
from elementsBody import *
from neuralNetwork import NeuralNetwork
from weightedAverage import *
from progressBar import *
from multiprocessing import Process

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

sys.path.append('./tryOpenGL')
import mhxOpenGL
import colorBar, Arrow3D
from sphere3D import putSphere

from show_entire_body import entireBody
from decideVfByGeometry import *
from horizon import *


class SmoothEntireBody(object):

    def __init__(self, obj):
        if not isinstance(obj, ElementsBody):
            raise ValidationErr("error, the input args is not an object of class ElementsBody. ")
        """ elements' neighbors should include themself, 
            this is used in the match between dense nodes and elements """
        obj.get_eleNeighbor()
        for iele in range(len(obj.elements)):
            obj.eleNeighbor[iele].add(iele)
        self.obj = obj  

    ### ===================================================================== related to fitting
   
    def get_VfNets(self, preRefine=True):
        """ 
            identify which element has neural network for fitting of VF, 
            get the neural networks of these elements
        """
        eps = 1.e-6
        obj = self.obj
        if not hasattr(obj, "eleNeighbor"):
            obj.get_eleNeighbor()
        if not hasattr(obj, "eLen"):
            obj.get_eLen()

        nets = {}
        for iele in range(len(obj.elements)):
            if eps < obj.VF[iele] < 1.-eps:
                nets[iele] = NeuralNetwork(mod="phi", dm=2)
            elif obj.VF[iele] <= eps:
                hasNeighborOne = False
                for other in obj.eleNeighbor[iele]:
                    if obj.VF[other] >= 1.-eps:
                        hasNeighborOne = True
                        break
                if hasNeighborOne:
                    nets[iele] = NeuralNetwork(mod="phi", dm=2)
        
        """ densify the sample for fitting """
        denseSamples = {}  # keys: element number
                           # values: dense nodes' coordinates attatched to this element
        if preRefine:
            vx, vy = np.array([1., 0., 0.]) * obj.eLen, np.array([0., 1., 0.]) * obj.eLen
            for iele in range(len(obj.elements)):
                sameNeighbor = True
                for other in obj.eleNeighbor[iele]:
                    if obj.VF[other] != obj.VF[iele]:
                        sameNeighbor = False
                        break
                if not sameNeighbor:
                    denseSamples[iele] = []
                    ### get the dense nodes' coordinates for this element
                    for i in [-1./3., 0, 1./3.]:
                        for j in [-1./3., 0, 1./3.]:
                            denseSamples[iele].append(
                                np.array(obj.eleCenter(iele)) + i * vx + j * vy
                            )
        
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
        
        """ fit the nets """
        totalFitRegion = set()
        for iele in nets:
            xyzs, vals = [], []
            horizon = horizons(obj, iele)
            totalFitRegion |= horizon
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
            nets[iele].getFitting(  # anagolus to the learning process by gradient decent method
                xyzs, vals, 
                region_cen=obj.eleCenter(iele),
                loss_tolerance=0.4,
                innerLoops=1,  # whether fit multiple times in the inner loop and choose a net with lowest loss
                plotData=False, printData=False, 
                frozenClassifier=False,  
                    ### frozenClassifier, whether freeze the classier (possiotion and direction), 
                    ### so that the wieghts and bias at hidden layer remian unchanged 
                prematurelyBreak=False,  # whether prematurely break when the parameters nearly unchanged at a step
                refitTimes=10,  # refit by changing the initial values of weights
            )
        self.vfNets, self.totalFitRegion = nets, totalFitRegion
    

    def get_stressNets(self):
        obj = self.obj
        if not hasattr(self, "vfNets"):
            self.get_VfNets()
        stressNets = {}
        for iele in self.vfNets:
            stressNets[iele] = NeuralNetwork(mod="stress", dm=2)
            horizon = horizons(obj, iele)
            ### fit the net by stress values
            xyzs = [obj.eleCenter(_) for _ in horizon]
            vals = [obj.stress[_] for _ in horizon]
            initialWei = list(map(lambda x: x.data, self.vfNets[iele].net.parameters()))
            initialWei[-2] = -initialWei[-2]
            initialWei[-1][:] = sum(vals) / len(vals)
            print("(max(vals) - min(vals)) / 2. =", (max(vals) - min(vals)) / 2.)
            stressNets[iele].getFitting(
                xyzs, vals,
                lr=0.2,  
                region_cen=obj.eleCenter(iele),
                loss_tolerance=((max(vals) - min(vals)) / 2.)**2,
                initialWei=initialWei,
                innerLoops=1,  # whether fit multiple times in the inner loop and choose a net with lowest loss
                plotData=False, printData=False, 
                frozenClassifier=False,  
                    ### frozenClassifier, whether freeze the classier (possiotion and direction), 
                    ### so that the wieghts and bias at hidden layer remian unchanged 
                prematurelyBreak=False,  # whether prematurely break when the parameters nearly unchanged at a step
                refitTimes=10,  # refit by changing the initial values of weights
            )
        self.stressNets = stressNets


    def draw_fitting_field(self, 
                        fieldOption="VF",  # field can be VF or stress 
                        minVal=None, maxVal=None,
                        drawArrows=False, 
                        preRefine=True):
        """
            draw the patch (entire body as a patch)
        """
        obj = self.obj
        if fieldOption == "VF":
            field = obj.VF
            self.get_VfNets(preRefine=preRefine)
            nets = self.vfNets
        elif fieldOption == "stress":
            field = obj.stress
            self.get_stressNets()
            nets = self.stressNets
        else:
            raise ValidationErr("error, fieldOption can be either 'VF' or 'stress'")
        
        obj.get_outerFacetDic()
        obj.get_facialEdgeDic()
        
        minVal = min(field) if minVal == None else minVal
        maxVal = max(field) if maxVal == None else maxVal

        patchEles = {i for i in range(len(obj.elements))}
        facets, edges = {}, {}
        for facet in obj.outerFacetDic:
            ele = obj.outerFacetDic[facet]

            neighborNets = []
            if ele in self.totalFitRegion:
                horizon = horizons(obj, ele)
                for other in horizon:
                    if other in nets:
                        neighborNets.append(other)
                if len(neighborNets) > 0:
                    ### find the nearest element who has net
                    netE = min(
                        neighborNets,
                        key=lambda other:
                            sum((np.array(obj.eleCenter(other)) - np.array(obj.eleCenter(ele)))**2)
                    )
                    valueFunc = nets[netE].reasoning
                else:
                    raise ValueError("error, len(neighborNets) == 0")
            else:  # ele not in self.totalFitRegion
                valueFunc = lambda x: field[ele]

            averageZ = sum([obj.nodes[node][2] for node in facet]) / len(facet)
            if averageZ > 0:
                facets[facet] = {}
                ### densify or not
                if len(neighborNets) > 0:
                    denseOrder = 4
                else:
                    denseOrder = 1
                # sameNeighbor = True
                # for other in obj.eleNeighbor[ele]:
                #     if obj.VF[other] != obj.VF[ele]:
                #         sameNeighbor = False
                #         break
                # if not sameNeighbor:
                #     denseOrder = 4  # densify by 4 times
                # else:
                #     denseOrder = 1

                ### get the densified facets inside big facets
                facets[facet]["denseNodesCoo"], facets[facet]["facets"], outerFrames = facetDenseRegular(
                    np.array([obj.nodes[node] for node in facet]), order=denseOrder
                )
                facets[facet]["fieldVals"] = {}
                for node in facets[facet]["denseNodesCoo"]:
                    facets[facet]["fieldVals"][node] = float(valueFunc(facets[facet]["denseNodesCoo"][node]))
                for edge in outerFrames:
                    xyzs = tuple(sorted([
                        tuple(facets[facet]["denseNodesCoo"][edge[0]]), 
                        tuple(facets[facet]["denseNodesCoo"][edge[1]]),
                    ]))
                    edges[xyzs] = list(map(lambda x: float(valueFunc(x)) + 0.1, xyzs))  # hight of the edge, 
                                                                                        # represented by field value
        
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
                        gradients[tuple(obj.eleCenter(ele))] = nets[ele].getGradient(obj.eleCenter(ele))
        
        obj.ratio_draw, obj.regionCen = obj.ratioOfVisualize()
        
        Process(
            target=mhxOpenGL.showUp, 
            args=(self.__simple_draw_patch__, 
                  facets, edges, 
                  minVal, maxVal,
                  obj.regionCen,  # regionCen
                  gradients)
        ).start()
    

    ### ======================================================================== related to weighted average
    
    def get_average_horizons(self, spreadRange):
        """
            get the horizons for weighted average, 
            called by self.draw_average_field()
        """
        eps = 1.e-6
        obj = self.obj
        if not hasattr(obj, "eleNeighbor"):
            obj.get_eleNeighbor()
        if not hasattr(obj, "eLen"):
            obj.get_eLen()

        averageHorizons = {}
        for iele in range(len(obj.elements)):
            if eps < obj.VF[iele] < 1.-eps:
                averageHorizons[iele] = obj.findHorizon(iele, 
                                                        obj.inSphericalHorizon, 
                                                        spreadRange)
            elif obj.VF[iele] <= eps:
                hasNeighborOne = False
                for other in obj.eleNeighbor[iele]:
                    if obj.VF[other] >= 1.-eps:
                        hasNeighborOne = True
                        break
                if hasNeighborOne:
                    averageHorizons[iele] = obj.findHorizon(iele, 
                                                            obj.inSphericalHorizon, 
                                                            spreadRange)
        
        """ get the total horizon """
        totalHorizon = set()
        for iele in averageHorizons:
            totalHorizon |= averageHorizons[iele]

        self.averageHorizons, self.totalHorizon = averageHorizons, totalHorizon
  

    def draw_average_field(self, 
                        fieldOption="VF",  # field can be VF or stress 
                        minVal=None, maxVal=None,
                        drawArrows=False, 
                        drawSmooth=True):
        """
            use weighted average
            draw the patch (entire body as a patch)
        """
        spreadRange = float(input("\033[32;1m please give the spread range for weighted average. "
                                  "spreadRange = \033[0m"))
        self.get_average_horizons(spreadRange)
        obj = self.obj
        if fieldOption == "VF":
            field = obj.VF
            averageFunc = weighted_average
        elif fieldOption == "stress":
            field = obj.stress
            averageFunc = weighted_average_forStress
        else:
            raise ValidationErr("error, fieldOption can be either 'VF' or 'stress'")
        
        obj.get_outerFacetDic()
        obj.get_facialEdgeDic()
        
        minVal = min(field) if minVal == None else minVal
        maxVal = max(field) if maxVal == None else maxVal

        patchEles = {i for i in range(len(obj.elements))}
        facets, edges, gradients = {}, {}, {}
        for facet in obj.outerFacetDic:
            ele = obj.outerFacetDic[facet]

            if drawSmooth and ele in self.totalHorizon:
                valueFunc = averageFunc
            else:
                valueFunc = lambda *arges: field[ele]

            averageZ = sum([obj.nodes[node][2] for node in facet]) / len(facet)
            if averageZ > 0:
                facets[facet] = {}
                ### densify or not
                if ele in self.totalHorizon:
                    denseOrder = 4
                else:
                    denseOrder = 1

                ### get the densified facets inside big facets
                facets[facet]["denseNodesCoo"], facets[facet]["facets"], outerFrames = facetDenseRegular(
                    np.array([obj.nodes[node] for node in facet]), order=denseOrder
                )
                facets[facet]["fieldVals"] = {}
                for node in facets[facet]["denseNodesCoo"]:
                    ### values at the facet nodes
                    facets[facet]["fieldVals"][node] = valueFunc(
                        facets[facet]["denseNodesCoo"][node],  # xyz
                        ele,  # iele
                        obj,  # obj1
                        "normal_distri",  # func
                        "val",  # result
                        2,  # dimension
                        spreadRange,  # spreadRange
                    )
                for edge in outerFrames:
                    xyzs = tuple(sorted([
                        tuple(facets[facet]["denseNodesCoo"][edge[0]]), 
                        tuple(facets[facet]["denseNodesCoo"][edge[1]]),
                    ]))
                    ### hight of the edge, represented by field value
                    edges[xyzs] = list(map(lambda x: float(valueFunc(
                        x,  # xyz
                        ele,  # iele
                        obj,  # obj1
                        "normal_distri",  # func
                        "val",  # result
                        2,  # dimension
                        spreadRange
                    )) + 0.1, xyzs))  
        
            ### whether draw the arrows of gradients
            if drawArrows:
                eps = 0.01
                if obj.VF[ele] < 1.-eps:
                    neighborOne = False
                    for other in obj.eleNeighbor[ele]:
                        if obj.VF[other] > 1.-eps:
                            neighborOne = True
                            break
                    if neighborOne:  # self small, neighbor = 1
                        gradients[tuple(obj.eleCenter(ele))] = averageFunc(
                            xyz=obj.eleCenter(ele), 
                            iele=ele, obj1=obj, func="normal_distri",
                            result="grad", 
                            spreadRange=spreadRange, 
                        )
        
        obj.ratio_draw, obj.regionCen = obj.ratioOfVisualize()
        
        Process(
            target=mhxOpenGL.showUp, 
            args=(self.__simple_draw_patch__, 
                  facets, edges, 
                  minVal, maxVal,
                  obj.regionCen,  # regionCen
                  gradients)
        ).start()

    ### ============================================================================= related to patch draw

    def __simple_draw_patch__(self, facets, edges, 
                              minVal, maxVal, 
                              regionCen,
                              gradients):
        obj = self.obj
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
                        (VF - regionCen[2]) * obj.ratio_draw
                    )
        glEnd()

        ### ----------------------------------- draw the element edges
        glLineWidth(3.)
        red, green, blue = 0.4, 0.4, 0.4
        glColor4f(red, green, blue, 1.0)
        glMaterialfv(GL_FRONT, GL_AMBIENT, [red, green, blue])
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [red, green, blue])
        glMaterialfv(GL_FRONT, GL_SPECULAR, [red, green, blue])
        glMaterialfv(GL_FRONT, GL_EMISSION, [red, green, blue])
        glBegin(GL_LINES)
        for edge in edges:
            glVertex3f((edge[0][0] - regionCen[0]) * obj.ratio_draw, 
                       (edge[0][1] - regionCen[1]) * obj.ratio_draw, 
                       (edges[edge][0] - regionCen[2]) * obj.ratio_draw)
            glVertex3f((edge[1][0] - regionCen[0]) * obj.ratio_draw, 
                       (edge[1][1] - regionCen[1]) * obj.ratio_draw, 
                       (edges[edge][1] - regionCen[2]) * obj.ratio_draw)
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
            h = 3.5 * obj.ratio_draw  # h = 1.5 * obj.ratio_draw  # use same length to visualize arrows

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
       

    ### ==================================================================================== related to dense nodes

    def draw_fitting_nodes(self, 
                        fieldOption="VF",  # field can be VF or stress 
                        minVal=None, maxVal=None,
                        preRefine=True):
        """
            draw the dense nodes of entire body,
            dense nodes' color shown by fitting-field value,
            fitting nodes aligned by lattice direction
        """
        obj = self.obj
        obj.get_eleNeighbor()
        obj.get_eLen()
        if fieldOption == "VF":
            field = obj.VF
            self.get_VfNets(preRefine=preRefine)
            nets = self.vfNets
        elif fieldOption == "stress":
            field = obj.stress
            self.get_stressNets()
            nets = self.stressNets
        else:
            raise ValidationErr("error, fieldOption can be either 'VF' or 'stress'")
        
        minVal = min(field) if minVal == None else minVal
        maxVal = max(field) if maxVal == None else maxVal

        denseNodes = []
        for ele in range(len(obj.elements)):
            if ele in self.totalFitRegion:
                neighborNets = []
                horizon = horizons(obj, ele)
                for other in horizon:
                    if other in nets:
                        neighborNets.append(other)
                if len(neighborNets) > 0:
                    ### find the nearest element who has net
                    netE = min(
                        neighborNets,
                        key=lambda other:
                            sum((np.array(obj.eleCenter(other)) - np.array(obj.eleCenter(ele)))**2)
                    )
                    valueFunc = nets[netE].reasoning
                else:
                    raise ValueError("error, len(neighborNets) == 0")
            else:  # ele not in self.totalFitRegion
                valueFunc = lambda x: field[ele]
            
            ### densify or not
            sameNeighbor = True
            for other in obj.eleNeighbor[ele]:
                if abs(obj.VF[other] - obj.VF[ele]) / (abs(obj.VF[ele]) + 1.e-8) > 1.e-3:
                    sameNeighbor = False
                    break
            
            ### get the denseNodes' positions and values, 
            ### where dense nodes are aligned by lattice direction
            if sameNeighbor:
                denseNodes.append({
                    "pos": obj.eleCenter(ele),
                    "val": float(valueFunc(obj.eleCenter(ele))),
                })
                # denseNodes[tuple(obj.eleCenter(ele))] = float(valueFunc(obj.eleCenter(ele)))
            else:
                for i in [-1, 0, 1]:
                    for j in [-1, 0, 1]:
                        position = \
                            np.array(obj.eleCenter(ele)) + \
                            np.array([i / 3. * obj.eLen, j / 3. * obj.eLen, 0.])
                        denseNodes.append({
                            "pos": position,
                            "val": float(valueFunc(position)),
                        })
                        # denseNodes[tuple(position)] = float(valueFunc(position))
        
        obj.ratio_draw, obj.regionCen = obj.ratioOfVisualize()
        print("\033[32;1m {} = \033[40;33;1m{} \033[0m".format("len(denseNodes)", len(denseNodes)))
        
        Process(
            target=mhxOpenGL.showUp, 
            args=(self.__draw_nodes__, 
                denseNodes, [], 
                minVal, maxVal,
                obj.regionCen,  # regionCen
                )
        ).start()


    def draw_regularDenseNodes_localDense(self, 
                        fieldOption="VF",  # field can be VF or stress 
                        minVal=None, maxVal=None,
                        preRefine=True):
        """
            draw the nodes of entire body, dense nodes near interface
            dense nodes' color shown by fitting-field value,
            fitting nodes aligned by lattice direction
        """
        obj = self.obj
        obj.get_eleNeighbor()
        obj.get_eLen()
        self.getRegularDenseNodes()
        if fieldOption == "VF":
            field = obj.VF
            self.get_VfNets(preRefine=preRefine)
            nets = self.vfNets
        elif fieldOption == "stress":
            field = obj.stress
            self.get_stressNets()
            nets = self.stressNets
        else:
            raise ValidationErr("error, fieldOption can be either 'VF' or 'stress'")
        
        minVal = min(field) if minVal == None else minVal
        maxVal = max(field) if maxVal == None else maxVal

        bigNodes, smallNodes = [], []
        for ele in range(len(obj.elements)):
            if ele in self.totalFitRegion:
                neighborNets = []
                horizon = horizons(obj, ele)
                for other in horizon:
                    if other in nets:
                        neighborNets.append(other)
                if len(neighborNets) > 0:
                    ### find the nearest element who has net
                    netE = min(
                        neighborNets,
                        key=lambda other:
                            sum((np.array(obj.eleCenter(other)) - np.array(obj.eleCenter(ele)))**2)
                    )
                    valueFunc = nets[netE].reasoning
                else:
                    raise ValueError("error, len(neighborNets) == 0")
            else:  # ele not in self.totalFitRegion
                valueFunc = lambda x: field[ele]
            
            ### densify or not
            sameNeighbor = True
            for other in obj.eleNeighbor[ele]:
                if abs(obj.VF[other] - obj.VF[ele]) / (abs(obj.VF[ele]) + 1.e-8) > 1.e-3:
                    sameNeighbor = False
                    break
            
            ### get the denseNodes' positions and values, 
            ### where dense nodes are aligned by lattice direction
            if sameNeighbor:
                bigNodes.append({
                    "pos": obj.eleCenter(ele),
                    "val": float(valueFunc(obj.eleCenter(ele))),
                })
            else:
                for i, j in self.ele_regularDenseNodes[ele]:
                    smallNodes.append({
                        "pos": np.append(self.denseNodes[i, j], 0.),
                        "val": float(valueFunc(self.denseNodes[i, j])),
                    })
        
        obj.ratio_draw, obj.regionCen = obj.ratioOfVisualize()
        print("\033[32;1m {} = \033[40;33;1m{} \033[0m".format("len(nodes)", len(bigNodes) + len(smallNodes)))
        
        Process(
            target=mhxOpenGL.showUp, 
            args=(self.__draw_nodes__, 
                smallNodes, bigNodes, 
                minVal, maxVal,
                obj.regionCen,  # regionCen
                )
        ).start()


    def getRegularDenseNodes(self):
        if not hasattr(self, "denseNodes"):
            ### find the element at the bottom left corner
            obj = self.obj

            bottomLeft = min(
                [i for i in range(len(obj.elements))], 
                key=lambda j: obj.eleCenter(j)[0] + obj.eleCenter(j)[1]
            )

            minX = min([obj.nodes[_][0] for _ in obj.nodes])
            maxX = max([obj.nodes[_][0] for _ in obj.nodes])
            minY = min([obj.nodes[_][1] for _ in obj.nodes])
            maxY = max([obj.nodes[_][1] for _ in obj.nodes])

            print("\033[40;33;1m {} {} \033[0m".format("bottom left element =", bottomLeft + 1))
            eLen = obj.getVolumes(eleNum=bottomLeft) ** (1. / 3.)

            ### compute the step between two dense nodes
            denseN = int(input("\033[35;1m {} \033[0m".format(
                "how many times do you want to densify on each diemnesion? times = "
            )))
            step = eLen / denseN
            denseNodesSize = [int((maxX - minX) / step), int((maxY - minY) / step)]
            ele_regularDenseNodes = {}

            ### get the coordinates of the starting node
            startCoord = np.array([minX, minY]) + np.array([step/2., step/2.])

            ### =============== use dynamic programming (DP) to get the dense nodes
            denseNodes = np.zeros((*denseNodesSize, len(startCoord)), dtype=startCoord.dtype)
            denseNodes[0][0] = startCoord
            denseNodes_ele = -np.ones((*denseNodesSize, ), dtype=int)
            denseNodes_ele[0][0] = bottomLeft
            denseNodesGraph = {(0, 0): set(), }
            ### the boundary condition of DP (dense nodes on bottom line and left line)
            for j in range(1, denseNodesSize[1]):
                denseNodes[0][j] = startCoord + np.array([j * step, 0.])
                # graph_double_link(denseNodesGraph, node0=(0, j-1), node1=(0, j))
                denseNodes_ele[0, j] = min(
                    obj.eleNeighbor[denseNodes_ele[0, j-1]],
                    key=lambda iele: 
                        sum((np.array(obj.eleCenter(iele))[:2] - denseNodes[0][j]) ** 2)
                )
            for i in range(1, denseNodesSize[0]):
                denseNodes[i][0] = startCoord + np.array([0., j * step])
                # graph_double_link(denseNodesGraph, node0=(i-1, 0), node1=(i, 0))
                denseNodes_ele[i, 0] = min(
                    obj.eleNeighbor[denseNodes_ele[i-1, 0]],
                    key=lambda iele: 
                        sum((np.array(obj.eleCenter(iele))[:2] - denseNodes[i][0]) ** 2)
                )
            ### get coordinates and link relations (graph) of the dense nodes
            nodesNum = (denseNodesSize[0] - 1) * (denseNodesSize[1] - 1)
            count = 0
            for j in range(1, denseNodesSize[1]):
                for i in range(1, denseNodesSize[0]):
                    count += 1
                    if count % 100 == 0:
                        progressBar_percentage(count / nodesNum * 100.)
                    denseNodes[i][j] = startCoord + np.array([j * step, i * step])
                    theEle = min(
                        obj.eleNeighbor[denseNodes_ele[i-1, j]] | obj.eleNeighbor[denseNodes_ele[i, j-1]],
                        key=lambda iele:
                            sum((np.array(obj.eleCenter(iele))[:2] - denseNodes[i][j]) ** 2)
                    )
                    denseNodes_ele[i, j] = theEle
                    if theEle in ele_regularDenseNodes:
                        ele_regularDenseNodes[theEle].append((i, j))
                    else:
                        ele_regularDenseNodes[theEle] = [(i, j), ]
            print("")  # break line for the progress bar
            self.denseNodes, self.denseNodes_ele, self.ele_regularDenseNodes = denseNodes, denseNodes_ele, ele_regularDenseNodes
            return denseNodes, denseNodes_ele, ele_regularDenseNodes


    def __draw_nodes__(self, smallNodes, bigNodes, minVal, maxVal, regionCen):
        obj = self.obj
        ### ----------------------------------- draw the dense nodes
        for node in smallNodes:
            red, green, blue = colorBar.getColor((node["val"] - minVal) / (maxVal - minVal))
            glColor4f(red, green, blue, 1.0)
            glMaterialfv(GL_FRONT, GL_AMBIENT, [red, green, blue])
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [red, green, blue])
            glMaterialfv(GL_FRONT, GL_SPECULAR, [red, green, blue])
            glMaterialfv(GL_FRONT, GL_EMISSION, [red, green, blue])

            radius = 0.1
            newPos = np.array([node["pos"][0], node["pos"][1], node["pos"][2] + radius])  # pos[2] += radius
            newPos -= np.array(regionCen)
            newPos, radius = newPos * obj.ratio_draw, radius * obj.ratio_draw
            putSphere(newPos, radius, [red, green, blue], resolution=10)

        for node in bigNodes:
            red, green, blue = colorBar.getColor((node["val"] - minVal) / (maxVal - minVal))
            glColor4f(red, green, blue, 1.0)
            glMaterialfv(GL_FRONT, GL_AMBIENT, [red, green, blue])
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [red, green, blue])
            glMaterialfv(GL_FRONT, GL_SPECULAR, [red, green, blue])
            glMaterialfv(GL_FRONT, GL_EMISSION, [red, green, blue])

            radius = 0.2
            newPos = np.array([node["pos"][0], node["pos"][1], node["pos"][2] + radius])  # pos[2] += radius
            newPos -= np.array(regionCen)
            newPos, radius = newPos * obj.ratio_draw, radius * obj.ratio_draw
            putSphere(newPos, radius, [red, green, blue], resolution=10)
        
        ### ---------------------------------------------------------------
        glutSwapBuffers()  # 切换缓冲区，以显示绘制内容


def averageVF(obj, iele, spreadRange=1.5):
    """
        get the average VF of each element of this object
    """
    if not isinstance(obj, ElementsBody):
        raise ValidationErr("error, not isinstance(obj, ElementsBody)")
    if not hasattr(obj, "averageVF"):
        obj.averageVF = {}
    
    if iele in obj.averageVF:
        return obj.averageVF[iele]
    else:
        obj.averageVF[iele] = weighted_average(
            xyz=np.array(obj.eleCenter(iele)), iele=iele,
            obj1=obj, 
            func="normal_distri", 
            spreadRange=spreadRange,
        )
        return obj.averageVF[iele]


if __name__ == "__main__":

    inpFile = input("\033[40;35;1m{}\033[0m".format(
        "please give the .inp file name (include the path): "
    ))
    obj1 = ElementsBody(
        *readInp(fileName=inpFile)
    )

    ### get the VF of each element of this object
    decideVfByGeometry(obj1, mod="constrainedSharp")

    # dataFile = input("\033[40;35;1m{}\033[0m".format(
    #     "please give the data file name (include the path): "
    # ))
    # dataFrame = readDataFrame(fileName=dataFile)
    # frame = int(input("which frame do you want? frame = "))
    # obj1.VF = dataFrame["SDV210_frame{}".format(frame)]
    # obj1.stress = dataFrame["SDV212_frame{}".format(frame)]
    
    ### get the maximum x and minimum, and obj1.ratio_draw
    decide_ratio_draw(obj1)
    
    ## show the entire body with fitting field
    field = SmoothEntireBody(obj1)  # ignite ---------------------------------------------------------
    # field.draw_fitting_field(fieldOption="VF", drawArrows=True, preRefine=True)
    # field.draw_average_field(fieldOption="VF", drawArrows=True, drawSmooth=True)
    field.draw_regularDenseNodes_localDense(fieldOption="VF")
    # field.draw_regularDenseNodes_localDense(fieldOption="stress", minVal=-550.)