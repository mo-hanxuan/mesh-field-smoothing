"""
    show a specific field, with locally densified at the place where gradient exists:
    with coarse edges to be visualized,

    user defind a field, with position as input, with 2 types of output:
        (1) a scalar
        (2) a vector indicate [R, G, B] to show the color 
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


class EntireField(object):

    def __init__(self, obj):
        if not isinstance(obj, ElementsBody):
            raise ValidationErr("error, the input args is not an object of class ElementsBody. ")
        """ elements' neighbors should include themself, 
            this is used in the match between dense nodes and elements """
        obj.get_eleNeighbor()
        for iele in range(len(obj.elements)):
            obj.eleNeighbor[iele].add(iele)
        self.obj = obj  

    ### ===================================================================== related to field defining
   
    def fieldVal(self, xyz=[0., 0., 0.], geometry="oneTwin"):
        """
            decide a smooth twin with long diffuse twin-tip
        """
        pos = np.array(xyz[:2])  # position
        return decide_twin_with_diffuse_tip(xyz=pos, 
                                           center=[20., 0.],
                                           longAxis0=4.5, longAxis1=24., 
                                           shortAxis0=1.5, shortAxis1=3., 
                                           angle=np.pi/3.)


    def fieldVal_RGB(self, xyz=[0., 0., 0.], geometry="oneTwin"):
        """
            decide a field with two diffuse twins crossing each other
        """
        pos = np.array(xyz[:2])  # position

        ### red color denotes the 1st twin
        red = decide_twin_with_diffuse_tip(xyz=pos, 
                                           center=[20., 0.],
                                           longAxis0=1.5, longAxis1=21., 
                                           shortAxis0=1.5, shortAxis1=3., 
                                           angle=np.pi/3.)
        ## blue color denotes another twin
        blue = decide_twin_with_diffuse_tip(xyz=pos, 
                                           center=[-4., -14.],  # [0., -12.]
                                           longAxis0=1.5, longAxis1=21., 
                                           shortAxis0=1.5, shortAxis1=3., 
                                           angle=0.)
        twinVF = red + blue
        if twinVF > 1.:
            return [red / twinVF, 0., blue / twinVF]
        else:
            return [red, 1. - red - blue, blue]


    def field_val_RGB_descrete(self, face="constrainedSharp"):
        obj = self.obj
        reds = decide_twin_with_descrete_vf(obj, center=[20., 0.],
                                            longAxis=15., shortAxis=3., 
                                            angle=np.pi/3.,
                                            face=face)
        blues = decide_twin_with_descrete_vf(obj, center=[-4., -14.],
                                            longAxis=15., shortAxis=3., 
                                            angle=0., 
                                            face=face)
        RGBs = np.zeros(shape=(len(obj.elements), 3), dtype=reds.dtype)
        RGBs[:, 0] = reds
        RGBs[:, 2] = blues
        for rgb in RGBs:
            totalVF = rgb[0] + rgb[2]
            if totalVF > 1.:
                rgb[0], rgb[2] = rgb[0] / totalVF, rgb[2] / totalVF
            rgb[1] = 1. - rgb[0] - rgb[2]
        obj.RGBs = RGBs
        return RGBs


    def draw_field(self, 
                fieldOption="VF",  # field can be VF or stress 
                minVal=None, maxVal=None,
                mod="RGB", 
                drawArrows=False, 
                preRefine=True):
        """
            draw the patch (entire body as a patch)
        """
        obj = self.obj
        
        obj.get_outerFacetDic()
        obj.get_facialEdgeDic()

        count_num_of_dense_facet = 0

        if mod == "RGB":
            fieldFunc = self.fieldVal_RGB
        else:
            fieldFunc = lambda x: float(self.fieldVal(x))

        facets, edges = {}, {}
        for facet in obj.outerFacetDic:
            averageZ = sum([obj.nodes[node][2] for node in facet]) / len(facet)
            
            if averageZ > 0:  # this uppper facet needs to be drawn
                facets[facet] = {}
                ### densify or not
                vals = [self.fieldVal(xyz=obj.nodes[node]) for node in facet]
                if all(vals) == True or any(vals) == False:
                    denseOrder = 1
                else:
                    count_num_of_dense_facet += 1
                    denseOrder = 4

                ### get the densified facets inside big facets
                facets[facet]["denseNodesCoo"], facets[facet]["facets"], outerFrames = facetDenseRegular(
                    np.array([obj.nodes[node] for node in facet]), order=denseOrder
                )
                facets[facet]["fieldVals"] = {}
                for node in facets[facet]["denseNodesCoo"]:
                    facets[facet]["fieldVals"][node] = fieldFunc(facets[facet]["denseNodesCoo"][node])
                for edge in outerFrames:
                    xyzs = tuple(sorted([
                        tuple(facets[facet]["denseNodesCoo"][edge[0]]), 
                        tuple(facets[facet]["denseNodesCoo"][edge[1]]),
                    ]))
                    edges[xyzs] = [0.5, 0.5]
                    # edges[xyzs] = list(map(lambda x: fieldFunc(x) + 0.1, xyzs))  # hight of the edge, 
                                                                                        # represented by field value
        
        print("\033[32;1m number of dense facets = \033[40;33;1m{}\033[0m".format(count_num_of_dense_facet))
        
        if mod == "val":
            if maxVal == None:
                maxVal = -float("inf")
                for facet in facets:
                    for node in facets[facet]["fieldVals"]:
                        maxVal = max(maxVal, facets[facet]["fieldVals"][node])
            if minVal == None:
                minVal = float("inf")
                for facet in facets:
                    for node in facets[facet]["fieldVals"]:
                        minVal = min(minVal, facets[facet]["fieldVals"][node])
            
        ### whether draw the arrows of gradients
        # gradients = {}  # key: position, value: gradient
        # if drawArrows:
        #     eps = 0.01
        #     for ele in patchEles:
        #         if obj.VF[ele] < 1.-eps:
        #             neighborOne = False
        #             for other in obj.eleNeighbor[ele]:
        #                 if obj.VF[other] > 1.-eps:
        #                     neighborOne = True
        #                     break
        #             if neighborOne:  # self small, neighbor = 1
        #                 gradients[tuple(obj.eleCenter(ele))] = nets[ele].getGradient(obj.eleCenter(ele))
        
        obj.ratio_draw, obj.regionCen = obj.ratioOfVisualize()
        
        Process(
            target=mhxOpenGL.showUp, 
            args=(self.__simple_draw_patch__, 
                  facets, edges, 
                  minVal, maxVal,
                  obj.regionCen,  # regionCen
                  {"red": {}, "blue": {}},  # gradients
                  )
        ).start()
    

    def draw_field_descrete(self, 
                        fieldOption="VF",  # field can be VF or stress 
                        minVal=None, maxVal=None,
                        mod="RGB", 
                        drawArrows=False, 
                        face="constrainedSharp"):
        """
            draw the patch (entire body as a patch)
        """
        obj = self.obj
        
        obj.get_outerFacetDic()
        obj.get_facialEdgeDic()

        RGBs = self.field_val_RGB_descrete(face=face)
        print("\033[32;1m RGBs = \n\033[40;33;1m{}\033[0m".format(RGBs))

        facets, edges = {}, {}
        for facet in obj.outerFacetDic:
            averageZ = sum([obj.nodes[node][2] for node in facet]) / len(facet)
            ele = obj.outerFacetDic[facet]
            if averageZ > 0:  # this uppper facet needs to be drawn
                facets[facet] = {}
                denseOrder = 1

                ### get the densified facets inside big facets
                facets[facet]["denseNodesCoo"], facets[facet]["facets"], outerFrames = facetDenseRegular(
                    np.array([obj.nodes[node] for node in facet]), order=denseOrder
                )
                facets[facet]["fieldVals"] = {}
                for node in facets[facet]["denseNodesCoo"]:
                    facets[facet]["fieldVals"][node] = RGBs[ele]
                for edge in outerFrames:
                    xyzs = tuple(sorted([
                        tuple(facets[facet]["denseNodesCoo"][edge[0]]), 
                        tuple(facets[facet]["denseNodesCoo"][edge[1]]),
                    ]))
                    edges[xyzs] = [0.5, 0.5]
                
        if mod == "val":
            if maxVal == None:
                maxVal = -float("inf")
                for facet in facets:
                    for node in facets[facet]["fieldVals"]:
                        maxVal = max(maxVal, facets[facet]["fieldVals"][node])
            if minVal == None:
                minVal = float("inf")
                for facet in facets:
                    for node in facets[facet]["fieldVals"]:
                        minVal = min(minVal, facets[facet]["fieldVals"][node])
        elif mod == "RGB":
            if maxVal == None:
                maxVal = [-float("inf") for _ in range(3)]
                for facet in facets:
                    for node in facets[facet]["fieldVals"]:
                        for i in range(3):
                            maxVal[i] = max(maxVal[i], facets[facet]["fieldVals"][node][i])
            if minVal == None:
                minVal = [float("inf") for _ in range(3)]
                for facet in facets:
                    for node in facets[facet]["fieldVals"]:
                        for i in range(3):
                            minVal[i] = min(minVal[i], facets[facet]["fieldVals"][node][i])
            
        ### whether draw the arrows of gradients
        if drawArrows:
            if mod == "val":
                gradients = {}  # key: position, value: gradient
                spreadRange = float(input("\033[32;1m spread range = \033[0m"))
                eps = 0.01
                for ele in obj.elements:
                    if obj.VF[ele] < 1.-eps:
                        neighborOne = False
                        for other in obj.eleNeighbor[ele]:
                            if obj.VF[other] > 1.-eps:
                                neighborOne = True
                                break
                        if neighborOne:  # self small, neighbor = 1
                            gradients[tuple(obj.eleCenter(ele))] = weighted_average(
                                xyz=obj.eleCenter(ele),
                                iele=ele, obj1=obj, 
                                func="normal_distri", 
                                result="grad", 
                                spreadRange=spreadRange,
                            )
            elif mod == "RGB":
                spreadRange = float(input("\033[32;1m spread range = \033[0m"))
                eps = 0.01

                redGradients = {}
                for ele in range(len(obj.elements)):
                    if RGBs[ele, 0] < 1.-eps:
                        neighborOne = False
                        for other in obj.eleNeighbor[ele]:
                            if RGBs[other, 0] > 1.-eps:
                                neighborOne = True
                                break
                        if neighborOne:  # self small, neighbor = 1
                            redGradients[tuple(obj.eleCenter(ele))] = weighted_average(
                                xyz=obj.eleCenter(ele),
                                iele=ele, obj1=obj, 
                                func="normal_distri", 
                                result="grad", 
                                spreadRange=spreadRange,
                                field=RGBs[:, 0],
                            )
                blueGradients = {}
                for ele in range(len(obj.elements)):
                    if RGBs[ele, 2] < 1.-eps:
                        neighborOne = False
                        for other in obj.eleNeighbor[ele]:
                            if RGBs[other, 2] > 1.-eps:
                                neighborOne = True
                                break
                        if neighborOne:  # self small, neighbor = 1
                            blueGradients[tuple(obj.eleCenter(ele))] = weighted_average(
                                xyz=obj.eleCenter(ele),
                                iele=ele, obj1=obj, 
                                func="normal_distri", 
                                result="grad", 
                                spreadRange=spreadRange,
                                field=RGBs[:, 2]
                            )
                gradients = {"red": redGradients, "blue": blueGradients}
        else:
            gradients = {"red": [], "blue": []}
        
        obj.ratio_draw, obj.regionCen = obj.ratioOfVisualize()
        
        Process(
            target=mhxOpenGL.showUp, 
            args=(self.__simple_draw_patch__, 
                  facets, edges, 
                  minVal, maxVal,
                  obj.regionCen,  # regionCen
                  gradients,  # gradients
                  mod,  # mod
                  )
        ).start()
    

    ### ============================================================================= related to patch draw

    def __simple_draw_patch__(self, facets, edges, 
                              minVal, maxVal, 
                              regionCen,
                              gradients, 
                              mod="RGB"):
        obj = self.obj
        ### ----------------------------------- draw the facets
        if mod == "val":
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
        elif mod == "RGB":
            glBegin(GL_QUADS)
            for bigFacet in facets:
                for smallFacet in facets[bigFacet]["facets"]:
                    for node in smallFacet:
                        red, green, blue = facets[bigFacet]["fieldVals"][node]
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
                       (edge[0][2] - regionCen[2]) * obj.ratio_draw)
            glVertex3f((edge[1][0] - regionCen[0]) * obj.ratio_draw, 
                       (edge[1][1] - regionCen[1]) * obj.ratio_draw, 
                       (edge[1][2] - regionCen[2]) * obj.ratio_draw)
        glEnd()

        ### ----------------------------------- draw the gradient arrows
        if mod == "val":
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
        elif mod == "RGB":
            ### =============================== red
            red, green, blue = 1., 0.2, 0.2
            glColor4f(red, green, blue, 1.0)
            glMaterialfv(GL_FRONT, GL_AMBIENT, [red, green, blue])
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [red, green, blue])
            glMaterialfv(GL_FRONT, GL_SPECULAR, [0., 1., 1.])
            glMaterialfv(GL_FRONT, GL_EMISSION, [0., 0., 0.])
            for position in gradients["red"]:
                r = np.array([*position[:2], 1.1]) - regionCen
                r *= obj.ratio_draw
                r = r.tolist()
                h = 3.5 * obj.ratio_draw  # h = 1.5 * obj.ratio_draw  # use same length to visualize arrows

                grad = gradients["red"][position]
                angle = np.degrees(np.arctan(grad[1] / grad[0]))
                if np.sign(grad[0]) == 0:
                    angle = np.sign(grad[1]) * 90.
                if grad[0] < 0:
                    angle -= 180.
                angle -= 180.
                Arrow3D.show(h, 0.05, angle, r)
            ### =============================== blue
            red, green, blue = 0.2, 0.2, 1.
            glColor4f(red, green, blue, 1.0)
            glMaterialfv(GL_FRONT, GL_AMBIENT, [red, green, blue])
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [red, green, blue])
            glMaterialfv(GL_FRONT, GL_SPECULAR, [0., 1., 1.])
            glMaterialfv(GL_FRONT, GL_EMISSION, [0., 0., 0.])
            for position in gradients["blue"]:
                r = np.array([*position[:2], 1.1]) - regionCen
                r *= obj.ratio_draw
                r = r.tolist()
                h = 3.5 * obj.ratio_draw  # h = 1.5 * obj.ratio_draw  # use same length to visualize arrows

                grad = gradients["blue"][position]
                angle = np.degrees(np.arctan(grad[1] / grad[0]))
                if np.sign(grad[0]) == 0:
                    angle = np.sign(grad[1]) * 90.
                if grad[0] < 0:
                    angle -= 180.
                angle -= 180.
                Arrow3D.show(h, 0.05, angle, r)
    
        ### ---------------------------------------------------------------
        glutSwapBuffers()  # 切换缓冲区，以显示绘制内容
       

def decide_twin_with_descrete_vf(obj, center, 
                                longAxis, shortAxis,
                                angle, 
                                face="constrainedSharp"): 
    if not isinstance(obj, ElementsBody):
        raise ValueError("error, not isinstance(obj, ElementsBody)")
    
    eigenVec1 = np.array([np.cos(angle), np.sin(angle)])
    eigenVec2 = np.array([-np.sin(angle), np.cos(angle)])
    mat = np.array([eigenVec1, eigenVec2]).transpose() @ \
            np.array([[1. / longAxis**2, 0.], 
                    [0., 1. / shortAxis**2]]) @ \
            np.array([eigenVec1, eigenVec2])
    
    if face == "constrainedSharp":
        denseSize = 8
        ### get the natural coordinates of insideNodes
        insideNodes = np.zeros((denseSize, denseSize, 2), dtype=np.array([1.]).dtype)
        for i in range(denseSize):
            for j in range(denseSize):
                insideNodes[i][j][0] = -1. + 1./denseSize + i * 2./denseSize
                insideNodes[i][j][1] = -1. + 1./denseSize + j * 2./denseSize
    
    obj.get_outerFacetDic()
    VF = np.zeros(shape=(len(obj.elements), ), dtype=np.float64)
    if face == "constrainedSharp":  # 0 < VF < 1 at the interface, VF is decided by integral
        for facet in obj.outerFacetDic:
            iele = obj.outerFacetDic[facet]
            averageZ = sum([obj.nodes[node][2] for node in facet]) / len(facet)
            if averageZ > 0:
                ### compute whether 4 nodes are twin or not
                nodesCoors = [np.array(obj.nodes[int(i)]) for i in facet]
                relativePositions = [coo[:2] - np.array(center[:2]) for coo in nodesCoors]
                nodeIsTwin = [pos @ mat @ pos for pos in relativePositions]
                nodeIsTwin = [val < 1. for val in nodeIsTwin]
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
                                shapeFunc2D(insideNodes[i][j]),
                                np.array(nodesCoors)
                            )
                            pos = globalCoor[:2] - np.array(center[:2])
                            if pos @ mat @ pos < 1.**2:
                                theVF += 1
                    VF[iele] = theVF / denseSize**2
    elif face == "simplifiedSharp":  # either 0 or 1 for VF
        for iele in range(len(obj.elements)):
            pos = np.array(obj.eleCenter(iele)[:2]) - np.array(center[:2])
            if pos @ mat @ pos < 1.**2:
                VF[iele] = 1.
            else:
                VF[iele] = 0.
    else:
        raise ValueError("error, args mod can either be 'constrainedSharp' or 'simplifiedSharp'")

    return VF
            

def decide_twin_with_diffuse_tip(xyz, center, 
                                longAxis0, longAxis1,  
                                shortAxis0, shortAxis1,  
                                angle):
    """
        decide a twin with diffuse twin tip, 
        longAxis0 and shortAxis0 are at the inside shell of the diffuse twin,
        longAxis1 and shortAxis1 are at the outside shell of the diffuse twin,

        at the diffuse interface from insde shell to the outside shell
        the vf smoothly changes from 1 to 0
    """
    ### judge whether the position is inside the twin
    eigenVec1 = np.array([np.cos(angle), np.sin(angle)])
    eigenVec2 = np.array([-np.sin(angle), np.cos(angle)])

    if longAxis0 > longAxis1:
        longAxis0, longAxis1 = longAxis1, longAxis0
    if shortAxis0 > shortAxis1:
        shortAxis0, shortAxis1 = shortAxis1, shortAxis0
    
    mat1 = np.array([eigenVec1, eigenVec2]).transpose() @ \
            np.array([[1. / longAxis1**2, 0.], 
                    [0., 1. / shortAxis1**2]]) @ \
            np.array([eigenVec1, eigenVec2])
    relativePosition = np.array(xyz)[:2] - center[:2]
    
    if relativePosition.transpose() @ mat1 @ relativePosition >= 1.**2:
        return 0.
    else:
        mat0 = np.array([eigenVec1, eigenVec2]).transpose() @ \
                np.array([[1. / longAxis0**2, 0.], 
                        [0., 1. / shortAxis0**2]]) @ \
                np.array([eigenVec1, eigenVec2])
        if relativePosition.transpose() @ mat0 @ relativePosition <= 1.**2:
            return 1.
        else:  # 0 < vf < 1
            l, r = 0., 1.  # left and right of vf
            while r - l > 1. / 128.:
                mid = (l + r) / 2.
                longAxis = longAxis0 + (longAxis1 - longAxis0) * mid
                shortAxis = shortAxis0 + (shortAxis1 - shortAxis0) * mid
                mat = np.array([eigenVec1, eigenVec2]).transpose() @ \
                        np.array([[1. / longAxis**2, 0.], 
                                [0., 1. / shortAxis**2]]) @ \
                        np.array([eigenVec1, eigenVec2])
                tmp = relativePosition.transpose() @ mat @ relativePosition
                if tmp < 1.:  # mid too big
                    r = mid
                elif tmp > 1:  # mid too small
                    l = mid 
                else:
                    break
            return 1. - mid
            

if __name__ == "__main__":

    inpFile = input("\033[40;35;1m{}\033[0m".format(
        "please give the .inp file name (include the path): "
    ))
    obj1 = ElementsBody(
        *readInp(fileName=inpFile)
    )

    ### get the VF of each element of this object
    # decideVfByGeometry(obj1, mod="constrainedSharp")

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
    field = EntireField(obj1)  # ignite ---------------------------------------------------------
    # field.draw_field()
    field.draw_field_descrete(drawArrows=False, face="constrainedSharp")