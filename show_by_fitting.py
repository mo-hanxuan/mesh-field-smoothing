"""
    show the dense nodes values by fitting
"""

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
import colorBar
from sphere3D import putSphere

from show_entire_body import entireBody
from decideVfByGeometry import *
from horizon import *


class regularDenseNodes(object):

    def __init__(self, obj):
        if not isinstance(obj, ElementsBody):
            raise ValidationErr("error, the input args is not an object of class ElementsBody. ")
        """ elements' neighbors should include themself, 
            this is used in the match between dense nodes and elements """
        obj.get_eleNeighbor()
        for iele in range(len(obj.elements)):
            obj.eleNeighbor[iele].add(iele)
        self.obj = obj


    def getDenseNodes(self):
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
                    # graph_double_link(denseNodesGraph, node0=(i-1, j), node1=(i, j))
                    # graph_double_link(denseNodesGraph, node0=(i, j-1), node1=(i, j))
                    denseNodes_ele[i, j] = min(
                        obj.eleNeighbor[denseNodes_ele[i-1, j]] | obj.eleNeighbor[denseNodes_ele[i, j-1]],
                        key=lambda iele:
                            sum((np.array(obj.eleCenter(iele))[:2] - denseNodes[i][j]) ** 2)
                    )
            print("")  # break line for the progress bar
            self.denseNodes, self.denseNodes_ele = denseNodes, denseNodes_ele
    

    def getDenseNodesVF(self, valueFunc="weighted_average"):
        if not hasattr(self, "denseNodesVF"):
            obj = self.obj
            if not hasattr(obj, "VF"):
                raise ValidationErr("error, input object has no attribute of VF! ")
            
            self.getDenseNodes()
            if valueFunc == "weighted_average":
                spreadRange = float(input("\033[35;1m {} \033[0m".format("spread range = ")))

            nodesVF = np.zeros(np.shape(self.denseNodes_ele), dtype=np.float64)
            count, nodesNum = 0, np.shape(nodesVF)[0] * np.shape(nodesVF)[1]
            for i in range(len(nodesVF)):
                for j in range(len(nodesVF[i])):
                    count += 1
                    if count % 100 == 0:
                        progressBar_percentage(count / nodesNum * 100.)
                    if valueFunc == "weighted_average":
                        nodesVF[i][j] = weighted_average(
                            xyz = np.array([*self.denseNodes[i][j], 0]), 
                            iele=int(self.denseNodes_ele[i][j]), 
                            obj1=obj, 
                            func="normal_distri", 
                            dimension=2,  # dimension for normal distribution
                            spreadRange=spreadRange,
                        )
                    else:
                        nodesVF[i][j] = obj.VF[self.denseNodes_ele[i][j]]
            self.denseNodesVF = nodesVF
        return self.denseNodesVF


    def getDenseNodesStress(self, valueFunc="weighted_average"):
        if not hasattr(self, "denseNodesStress"):
            obj = self.obj
            if not hasattr(obj, "stress"):
                raise ValidationErr("error, input object has no attribute of stress! ")
            
            self.getDenseNodes()
            spreadRange = float(input("\033[35;1m {} \033[0m".format("spread range = ")))

            nodesStress = np.zeros(np.shape(self.denseNodes_ele), dtype=np.float64)
            count, nodesNum = 0, np.shape(nodesStress)[0] * np.shape(nodesStress)[1]
            for i in range(len(nodesStress)):
                for j in range(len(nodesStress[i])):
                    count += 1
                    if count % 100 == 0:
                        progressBar_percentage(count / nodesNum * 100.)
                    if valueFunc == "weighted_average":
                        nodesStress[i][j] = weighted_average_forStress(
                            xyz = np.array([*self.denseNodes[i][j], 0]), 
                            iele=int(self.denseNodes_ele[i][j]), 
                            obj1=obj, 
                            func="normal_distri", 
                            dimension=2,  # dimension for normal distribution
                            spreadRange=spreadRange,
                        )
                    else:
                        nodesStress[i][j] = obj.stress[self.denseNodes_ele[i][j]]
            self.denseNodesStress = nodesStress
        return self.denseNodesStress
    

    def getDenseNodesVF_byEllip(self):
        """
            get the dense Nodes' VF by their coordinates, 
            use an elliptic shape incliusion, 
            inside is red, outside is blue, 
            smooth across the boundary, 
            boundary has thickness of 1 l0,  
        """
        longAxis, shortAxis = 20., 5.
        thickness = 1.5
        obj = self.obj
        self.getDenseNodes()
        denseNodes = self.denseNodes

        nodesVF = np.zeros(np.shape(self.denseNodes_ele), dtype=np.float64)
        count, nodesNum = 0, np.shape(nodesVF)[0] * np.shape(nodesVF)[1]
        for i in range(len(nodesVF)):
            for j in range(len(nodesVF[i])):
                if (denseNodes[i][j][0] / (longAxis - thickness/2))**2 \
                  + (denseNodes[i][j][1] / (shortAxis - thickness/2))**2 <= 1.**2:
                    nodesVF[i][j] = 1.
                elif (denseNodes[i][j][0] / (longAxis + thickness/2))**2 \
                  + (denseNodes[i][j][1] / (shortAxis + thickness/2))**2 >= 1.**2:
                    nodesVF[i][j] = 0.
                else:  # find the range located at the smooth boundary, use binary search
                    l, r = -thickness/2, thickness/2  # left end and right end of the range
                    while (r - l) > (thickness / 32.):
                        mid = (l + r) / 2.  # middle
                        tmp = (denseNodes[i][j][0] / (longAxis + mid))**2 \
                                + (denseNodes[i][j][1] / (shortAxis + mid))**2
                        if tmp < 1.:  # mid too big
                            r = mid
                        elif tmp > 1.:  # mid too small
                            l = mid
                        else:
                            break
                    nodesVF[i][j] = 1. - (mid / thickness + 0.5)

        self.denseNodesVF = nodesVF
        return self.denseNodesVF
    

    def get_VfNets(self):
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
        vx, vy = np.array([1., 0., 0.]) * obj.eLen, np.array([0., 1., 0.]) * obj.eLen
        denseSamples = {}  # keys: element number
                           # values: dense nodes' coordinates attatched to this element
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


    def getDenseNodesVfByFitValues(self):
        obj = self.obj
        if not hasattr(self, "vfNets"):
            self.get_VfNets()
        if not hasattr(self, "denseNodes"):
            self.getDenseNodes()
        
        denseNodesVF = np.zeros(np.shape(self.denseNodes_ele), dtype=np.float64)
        count, nodesNum = 0, np.shape(denseNodesVF)[0] * np.shape(denseNodesVF)[1]
        for i in range(len(denseNodesVF)):
            for j in range(len(denseNodesVF[i])):
                count += 1
                if count % 100 == 0:
                    progressBar_percentage(count / nodesNum * 100.)
                iele = self.denseNodes_ele[i][j]
                if iele in self.totalFitRegion:
                    horizon = horizons(obj, iele)
                    neighborNets = []
                    for ele in horizon:
                        if ele in self.vfNets:
                            neighborNets.append(ele)
                    if len(neighborNets) > 0:
                        ### find the nearest element who has net
                        netE = min(
                            neighborNets,
                            key=lambda ele:
                                sum((np.array(obj.eleCenter(ele)) - np.array(obj.eleCenter(iele)))**2)
                        )
                        denseNodesVF[i][j] = self.vfNets[netE].reasoning(self.denseNodes[i][j])
                    else:
                        raise ValueError("error, len(neighborNets) == 0")
                else:  # iele not in self.totalFitRegion
                    denseNodesVF[i][j] = obj.VF[iele]
        self.denseNodesVF = denseNodesVF
        

    def getDenseNodesStressByFitValues(self):
        obj = self.obj
        if not hasattr(self, "stressNets"):
            self.get_stressNets()
        if not hasattr(self, "denseNodes"):
            self.getDenseNodes()
        
        denseNodesStress = np.zeros(np.shape(self.denseNodes_ele), dtype=np.float64)
        count, nodesNum = 0, np.shape(denseNodesStress)[0] * np.shape(denseNodesStress)[1]
        for i in range(len(denseNodesStress)):
            for j in range(len(denseNodesStress[i])):
                count += 1
                if count % 100 == 0:
                    progressBar_percentage(count / nodesNum * 100.)
                iele = self.denseNodes_ele[i][j]
                if iele in self.totalFitRegion:
                    horizon = horizons(obj, iele)
                    neighborNets = []
                    for ele in horizon:
                        if ele in self.vfNets:
                            neighborNets.append(ele)
                    if len(neighborNets) > 0:
                        ### find the nearest element who has net
                        netE = min(
                            neighborNets,
                            key=lambda ele:
                                sum((np.array(obj.eleCenter(ele)) - np.array(obj.eleCenter(iele)))**2)
                        )
                        denseNodesStress[i][j] = self.stressNets[netE].reasoning(self.denseNodes[i][j])
                    else:
                        raise ValueError("error, len(neighborNets) == 0")
                else:  # iele not in self.totalFitRegion
                    denseNodesStress[i][j] = obj.stress[iele]
        self.denseNodesStress = denseNodesStress


    def drawDenseNodes(self, region_cen=None, 
                       funcVF=None, field="VF", valueFunc="weighted_average", 
                       maxVal=None, minVal=None):
        if field == "VF":
            if funcVF == None:
                self.getDenseNodesVF(valueFunc=valueFunc)
            else:
                funcVF()
            maxVal = self.denseNodesVF.max() if maxVal == None else maxVal
            minVal = self.denseNodesVF.min() if minVal == None else minVal
        else:
            if funcVF == None:
                self.getDenseNodesStress(valueFunc=valueFunc)
            else:
                self.getDenseNodesStressByFitValues()
            maxVal = self.denseNodesStress.max() if maxVal == None else maxVal
            minVal = self.denseNodesStress.min() if minVal == None else minVal

        if region_cen == None:
            region_cen = [0., 0.]
        self.region_cen = region_cen
        Process(target=mhxOpenGL.showUp, 
                args=(self.__drawDenseNodes__, field, maxVal, minVal)).start()
        # mhxOpenGL.showUp(self.__drawDenseNodes__, field, maxVal, minVal)
        
    
    def __drawDenseNodes__(self, field, maxVal, minVal):
        denseNodes = self.denseNodes
        obj = self.obj
        region_cen = self.region_cen

        if field == "VF":
            theField = self.denseNodesVF
        else:
            theField = self.denseNodesStress

        glClearColor(0.7, 0.7, 0.7, 1.0)
        
        ### ----------------------------------- draw the dense nodes
        for i in range(len(denseNodes)):
            for j in range(len(denseNodes[i])):

                red, green, blue = colorBar.getColor((theField[i][j] - minVal) / (maxVal - minVal))
                # red, green, blue = 0.6, 0.6, 0.6
                glColor4f(red, green, blue, 1.0)
                glMaterialfv(GL_FRONT, GL_AMBIENT, [red, green, blue])
                glMaterialfv(GL_FRONT, GL_DIFFUSE, [red, green, blue])
                glMaterialfv(GL_FRONT, GL_SPECULAR, [red, green, blue])
                glMaterialfv(GL_FRONT, GL_EMISSION, [red, green, blue])

                radius = 0.1
                pos = [*denseNodes[i][j], radius]
                pos[2] += radius
                pos -= np.array([*region_cen, 0.])
                # pos[0] += 12.  # dense nodes deviate from region_cen of visualization
                pos, radius = pos * obj.ratio_draw, radius * obj.ratio_draw
                putSphere(pos, radius, [red, green, blue], resolution=10)
        
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
    decideVfByGeometry(obj1)

    dataFile = input("\033[40;35;1m{}\033[0m".format(
        "please give the data file name (include the path): "
    ))
    dataFrame = readDataFrame(fileName=dataFile)
    frame = int(input("which frame do you want? frame = "))
    obj1.VF = dataFrame["SDV210_frame{}".format(frame)]
    obj1.stress = dataFrame["SDV212_frame{}".format(frame)]

    ### the shape of an elliptic
    longAxis, shortAxis = 20., 5.
    
    ### get the maximum x and minimum, and obj1.ratio_draw
    decide_ratio_draw(obj1)
    
    ### show the entire body
    # obj1.region_cen = [0., 0.]
    # body1 = entireBody(obj1)  # ignite ---------------------------------------------------------
    # body1.show_entire_body()

    ### get the object of dense nodes
    allDenseNodes = regularDenseNodes(obj1)

    ### draw the dense nodes with color of VF
    # allDenseNodes.drawDenseNodes()
    # allDenseNodes.drawDenseNodes(field="VF", valueFunc="direct")
    allDenseNodes.drawDenseNodes(field="VF", funcVF=allDenseNodes.getDenseNodesVfByFitValues)
    allDenseNodes.drawDenseNodes(field="stress", funcVF=allDenseNodes.getDenseNodesStressByFitValues, maxVal=700., minVal=-550.)
    # allDenseNodes.drawDenseNodes(field="stress", valueFunc="direct", maxVal=700., minVal=-550.)