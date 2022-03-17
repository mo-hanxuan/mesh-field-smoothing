"""
nonlinear fitting

before fitting, the sample points are densified,
then the dense samples are taken to fitting
"""

import numpy as np
import matplotlib.pyplot as plt
import multiprocessing, time

import sys

from torch._C import InterfaceType
from decideVfByGeometry import decideVfByGeometry
from decide_ratio_draw import decide_ratio_draw
from elementsBody import *
from scipy.interpolate import griddata
from collections import OrderedDict

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


class entireBody(object):
    def __init__(self, obj1) -> None:
        if not isinstance(obj1, ElementsBody):
            raise ValidationErr("error, not isinstance(obj1, ElementsBody) ")
        self.obj1 = obj1
        self.region_cen = obj1.region_cen

    def show_entire_body(self,):
        """
            show the entire body,
            along with its densified nodes near interface,
            and the gradient vectors at interface
        """
        obj1 = self.obj1
        eps = 1.e-3  # a value nearly 0
        if not hasattr(obj1, "eleNeighbor"):
            obj1.get_eleNeighbor()
        elif obj1.eleNeighbor == None:
            obj1.get_eleNeighbor()
        
        ### get the interface element
        interfaceOut, interfaceIn = set(), set()  # the inside and outside of the interface
        for ele in range(len(obj1.elements)):
            if obj1.VF[ele] < (1. - eps):  # < 1
                flag = False
                for ele2 in obj1.eleNeighbor[ele]:
                    if obj1.VF[ele2] >= (1. - eps):  # ≈ 1
                        flag = True
                        break
                if flag:
                    interfaceOut.add(ele)
            else: # ≈ 1
                for ele2 in obj1.eleNeighbor[ele]:
                    flag = False
                    if obj1.VF[ele2] < (1. - eps):  # < 1
                        flag = True
                        break
                if flag:
                    interfaceIn.add(ele)
        print("\033[40;32;1m{} \033[40;33m{}\033[0m".format(
            "len(interfaceOut) =", len(interfaceOut)
        ))

        ### get the volumes and eLen of this object
        if not hasattr(obj1, "volumes"):
            obj1.getVolumes(0)
        obj1.eLen = obj1.getVolumes(0) ** (1./3.)
        print("\033[40;31;1m{} \033[40;33;1m{}\033[0m".format(
            "(obj1.eLen, computed by average ele volume) ->\n"
            "               characteristic element length =", obj1.eLen
        ))

        ### get all elements center coordinates
        if len(obj1.eCen) == 0:
            for i in range(len(obj1.elements)):
                obj1.eleCenter(i)

        set_sparse = set(i for i in range(len(obj1.elements)))
        set_dense = interfaceIn | interfaceOut
        set_sparse -= set_dense

        denseNodes = {"pos":[], "eles":[], "vals":[]}
        grads = {"pos":[], "grads":[]}
        vx = np.array([1., 0., 0.]) * float(obj1.eLen)  # base vector on x direction
        vy = np.array([0., 1., 0.]) * float(obj1.eLen)  # base vector on y direction

        spreadRange = float(input("\033[35;1m {} \033[0m".format("spreadRange =")))
        
        print("\033[32m{}\033[0m".format("now get all dense nodes:"))
        count = 0
        for ele in set_dense:
            count += 1
            if count % 5 == 0:
                progressBar_percentage((count / len(set_dense)) * 100.)
            for i in range(-1, 2):
                for j in range(-1, 2):
                    pos = np.array(obj1.eleCenter(ele)) \
                        + (1./3.) * i * vx \
                        + (1./3.) * j * vy
                    denseNodes["pos"].append(pos)
                    denseNodes["vals"].append(
                        weighted_average(
                            pos, ele, obj1, 
                            func="normal_distri",
                            result='val',  # return value or return grad
                            dimension=2,  # dimension for normal distribution
                            spreadRange=spreadRange,
                        )
                    )
        print("\033[32m{}\033[0m".format("now get all gradient vectors:"))
        count = 0
        for ele in interfaceOut:
            count += 1
            if count % 5 == 0:
                progressBar_percentage((count / len(interfaceOut)) * 100.)
            pos = np.array(obj1.eleCenter(ele))
            grads["pos"].append(pos)
            grads["grads"].append(
                weighted_average(
                    pos, ele, obj1, 
                    func="normal_distri",
                    result='grad',  # return value or return grad
                    dimension=2,  # dimension for normal distribution
                    spreadRange=spreadRange,
                )
            )

        nonlinear, w = 1, 1
        X, Y, Z, idx, X1, Y1, Z1, \
        Xden, Yden, Zden = XYZforGL.get(
            obj1, w, [0, 0],
            nonlinear,
            dense_region=set(),
            sparse_region=set(i for i in range(len(obj1.elements))),
            dense_edges=False,
            show_original_field=True,
        )
        Zmax, Zmin = Z.max(), Z.min()

        self.dataPack = (
            X, Y, Z, idx, X1, Y1, Z1,
            Xden, Yden, Zden,
            Zmax, Zmin,
            {"pos": []},  # denseNodes
            grads, 
        )

        print("haha, hello world!")
        mhxOpenGL.showUp(self.draw_of_show_entire_body)

    
    def simple_draw_body(self, field, 
                         minVal=None, maxVal=None,
                         brightness=1., 
                         denseEles=None, 
                         averageFunc=weighted_average):
        obj = self.obj1
        obj.get_outerFacetDic()
        obj.get_facialEdgeDic()

        minVal = min(field) if minVal == None else minVal
        maxVal = max(field) if maxVal == None else maxVal

        if denseEles:
            denseNodes = {"pos":[], "vals":[]}
            obj.eLen = obj.getVolumes(0) ** (1./3.)
            vx = np.array([1., 0., 0.]) * float(obj.eLen)  # base vector on x direction
            vy = np.array([0., 1., 0.]) * float(obj.eLen)  # base vector on y direction
            vz = np.array([0., 0., 1.]) * float(obj.eLen)  # base vector on z direction

            spreadRange = float(input("\033[35;1m {} \033[0m".format("spreadRange =")))
            print("\033[32m{}\033[0m".format("now get all dense nodes:"))
            count = 0
            for ele in denseEles:
                count += 1
                if count % 5 == 0:
                    progressBar_percentage((count / len(denseEles)) * 100.)
                for i in range(-1, 2):
                    for j in range(-1, 2):
                        pos = np.array(obj.eleCenter(ele)) \
                            + (1./3.) * i * vx \
                            + (1./3.) * j * vy \
                            + vz
                        denseNodes["pos"].append(pos)
                        denseNodes["vals"].append(
                            averageFunc(
                                pos, ele, obj, 
                                func="normal_distri",
                                result='val',  # return value or return grad
                                dimension=2,  # dimension for normal distribution
                                spreadRange=spreadRange,
                            )
                        )
        else:
            denseNodes = []

        mhxOpenGL.showUp(self.__simple_draw_body__, field, minVal, maxVal, brightness, denseNodes)
    

    def __simple_draw_body__(self, field, 
                             minVal, maxVal,  # minVal and maxVal of field value
                             brighness=1.,  # brightness of the field
                             denseNodes=None):
        """
            simple draw the outer face of the body, 
            with color shown by field values
        """
        obj = self.obj1

        ### ----------------------------------- draw the dense Nodes
        if denseNodes:
            for i in range(len(denseNodes["pos"])):
                val = denseNodes["vals"][i]
                # print("denseNodes['vals'][i] = ", denseNodes["vals"][i])
                red, green, blue = colorBar.getColor((val - minVal) / (maxVal - minVal))
                glColor4f(red, green, blue, 1.0)
                glMaterialfv(GL_FRONT, GL_AMBIENT, [red, green, blue])
                glMaterialfv(GL_FRONT, GL_DIFFUSE, [red, green, blue])
                glMaterialfv(GL_FRONT, GL_SPECULAR, [red, green, blue])
                glMaterialfv(GL_FRONT, GL_EMISSION, [red, green, blue])

                radius = 0.11
                pos = denseNodes["pos"][i]
                pos -= np.array([*self.region_cen, 0.])
                pos, radius = pos * obj1.ratio_draw, radius * obj1.ratio_draw
                putSphere(pos, radius, [red, green, blue], resolution=10)

        ### ----------------------------------- draw the facets
        glBegin(GL_QUADS)
        for facet in obj.outerFacetDic:
            VF = field[obj.outerFacetDic[facet]]
            color = (VF - minVal) / (maxVal - minVal)
            red, green, blue = colorBar.getColor(color)
            glColor4f(red * brighness, green * brighness, blue * brighness, 1.0)
            glMaterialfv(GL_FRONT, GL_AMBIENT, [red, green, blue])
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [red, green, blue])
            glMaterialfv(GL_FRONT, GL_SPECULAR, [red, green, blue])
            glMaterialfv(GL_FRONT, GL_EMISSION, [red, green, blue])
            for node in facet:
                glVertex3f(obj.nodes[node][0] * obj.ratio_draw, 
                           obj.nodes[node][1] * obj.ratio_draw, 
                           obj.nodes[node][2] * obj.ratio_draw)
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
        for edge in obj.facialEdgeDic:
            glVertex3f(obj.nodes[edge[0]][0] * obj.ratio_draw, 
                       obj.nodes[edge[0]][1] * obj.ratio_draw, 
                       obj.nodes[edge[0]][2] * obj.ratio_draw)
            glVertex3f(obj.nodes[edge[1]][0] * obj.ratio_draw, 
                       obj.nodes[edge[1]][1] * obj.ratio_draw, 
                       obj.nodes[edge[1]][2] * obj.ratio_draw)
        glEnd()
        
        ### ---------------------------------------------------------------
        glutSwapBuffers()  # 切换缓冲区，以显示绘制内容


    def draw_of_show_entire_body(self):
        # coordinateLine.draw(lineWidth=3.,
        #                     lineLength=0.8)  # draw coordinate lines
        # --------------------------------- draw the element planes
        X, Y, Z, idx, X1, Y1, Z1, \
            Xden, Yden, Zden, \
            Zmax, Zmin, \
            denseNodes, grads, \
        = self.dataPack

        dense_edges = False
        obj1, region_cen = self.obj1, self.region_cen

        # ----------------------------------- draw the dense nodes
        for i in range(len(denseNodes["pos"])):

            red, green, blue = colorBar.getColor(denseNodes["vals"][i])
            # red, green, blue = 0.6, 0.6, 0.6
            glColor4f(red, green, blue, 1.0)
            glMaterialfv(GL_FRONT, GL_AMBIENT, [red, green, blue])
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [red, green, blue])
            glMaterialfv(GL_FRONT, GL_SPECULAR, [red, green, blue])
            glMaterialfv(GL_FRONT, GL_EMISSION, [red, green, blue])

            radius = 0.1
            pos = [denseNodes["pos"][i][0], denseNodes["pos"][i][1], radius]
            pos[2] += radius
            pos -= np.array([*self.region_cen, 0.])
            # pos[0] += 12.  # dense nodes deviate from region_cen of visualization
            pos, radius = pos * obj1.ratio_draw, radius * obj1.ratio_draw
            putSphere(pos, radius, [red, green, blue], resolution=10)
        # ---------------------------------------------------------------

        # ----------------------------------- draw the gradient vector
        red, green, blue = 1., 1., 1.
        glColor4f(red, green, blue, 1.0)
        glMaterialfv(GL_FRONT, GL_AMBIENT, [red, green, blue])
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [red, green, blue])
        glMaterialfv(GL_FRONT, GL_SPECULAR, [0., 1., 1.])
        glMaterialfv(GL_FRONT, GL_EMISSION, [0., 0., 0.])
        for i in range(len(grads["pos"])):
            r = np.array([grads["pos"][i][0], grads["pos"][i][1], 0.4])
            r[0:2] -= np.array(region_cen)
            r *= obj1.ratio_draw
            r = r.tolist()
            h = (np.array(grads["grads"][i]) ** 2).sum() ** (1./2.)
            h = 4. * obj1.eLen  # what if every grad vector use the the same length to visualize

            h *= obj1.ratio_draw

            angle = np.degrees(np.arctan(grads["grads"][i][1] / grads["grads"][i][0]))

            if np.sign(grads["grads"][i][0]) == 0:
                angle = np.sign(grads["grads"][i][1]) * 90.

            if grads["grads"][i][0] < 0:
                angle -= 180.
            angle -= 180.
            Arrow3D.show(h, 0.05, angle, r)
        # ---------------------------------------------------------------

        glBegin(GL_QUADS)
        for i in range(len(X)):
            color = (Z[i] - Zmin) / (Zmax - Zmin)
            # color = (obj1.rss[idx[i] - 1] - min(stress2)) / (max(stress2) - min(stress2))
            red, green, blue = colorBar.getColor(color)
            glColor4f(red * 1., green * 1., blue * 1., 1.0)
            glMaterialfv(GL_FRONT, GL_AMBIENT, [red, green, blue])
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [red, green, blue])
            glMaterialfv(GL_FRONT, GL_SPECULAR, [red, green, blue])
            glMaterialfv(GL_FRONT, GL_EMISSION, [red, green, blue])
            glVertex3f(X[i],
                    Y[i],
                    0.)
        glEnd()

        # ----------------------------------- draw the element edges
        if dense_edges == True:
            glLineWidth(2.)
            red, green, blue = 0.4, 0.4, 0.4
            glColor4f(red, green, blue, 1.0)
            glMaterialfv(GL_FRONT, GL_AMBIENT, [red, green, blue])
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [red, green, blue])
            glMaterialfv(GL_FRONT, GL_SPECULAR, [red, green, blue])
            glMaterialfv(GL_FRONT, GL_EMISSION, [red, green, blue])
            glBegin(GL_LINES)
            for i in range(len(Xden)):
                glVertex3f(Xden[i],
                        Yden[i],
                        0.)
            glEnd()

        glLineWidth(3.0)
        red, green, blue = 0.01, 0.01, 0.01
        glColor4f(red, green, blue, 1.0)
        glMaterialfv(GL_FRONT, GL_AMBIENT, [red, green, blue])
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [red, green, blue])
        glMaterialfv(GL_FRONT, GL_SPECULAR, [red, green, blue])
        glMaterialfv(GL_FRONT, GL_EMISSION, [red, green, blue])
        glBegin(GL_LINES)
        for i in range(len(X1)):
            glVertex3f(X1[i],
                    Y1[i],
                    0.)
        glEnd()

        # show the center point by a line
        glLineWidth(3.0)
        red, green, blue = 1., 1., 1.
        glColor4f(red, green, blue, 1.0)
        glMaterialfv(GL_FRONT, GL_AMBIENT, [red, green, blue])
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [red, green, blue])
        glMaterialfv(GL_FRONT, GL_SPECULAR, [red, green, blue])
        glMaterialfv(GL_FRONT, GL_EMISSION, [red, green, blue])
        glBegin(GL_LINES)
        glVertex3f(0., 0., 0.)
        glVertex3f(0., 0., 5.)
        glEnd()

        # ---------------------------------------------------------------
        glutSwapBuffers()  # 切换缓冲区，以显示绘制内容

    
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
    region_cen = [0., 0.]

    ### ================================= set the volume fraction for the elements
    decideVfByGeometry(obj1)
    
    # dataFile = input("\033[40;35;1m{}\033[0m".format(
    #     "please give the data file name (include the path): "
    # ))
    # dataFrame = readDataFrame(fileName=dataFile)
    # frame = int(input("which frame do you want? frame = "))
    # obj1.VF = dataFrame["SDV210_frame{}".format(frame)]
    # obj1.stress = dataFrame["SDV212_frame{}".format(frame)]
    
    obj1.region_cen = region_cen
    body1 = entireBody(obj1)  # ignite ---------------------------------------------------------
    
    ### show the body now
    body1.simple_draw_body(
        field=obj1.VF, 
    )
    # body1.simple_draw_body(
    #     field=obj1.stress, 
    #     minVal=-550.,
    #     brightness=0.6, 
    #     denseEles=denseEles, 
    #     averageFunc=weighted_average_forStress,
    # )

    ### get the element in the densified region
    obj1.get_eleNeighbor()
    denseEles = []
    eps = 1.e-6
    for iele in range(len(obj1.elements)):
        sameNeighbor = True
        for other in obj1.eleNeighbor[iele]:
            if abs(obj1.VF[other] - obj1.VF[iele]) > eps:
                sameNeighbor = False
                break
        if not sameNeighbor:
            denseEles.append(iele)
    print("\033[32;1m {} {} \033[0m".format("len(denseEles) =", len(denseEles)))
    

