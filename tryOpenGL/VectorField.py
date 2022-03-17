# -*- coding: utf-8 -*-

# -------------------------------------------
# draw the vector field of FEM odb file
# -------------------------------------------

import sys
# sys.path.append('../tryOpenGL')  # mhxOpenGL is in this path

import mhxOpenGL
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import math
import numpy as np

# ------------------------------------------------------import user-define modules
import colorBar
import Arrow3D
import coordinateLine
from Element import C3D8
from Element import Object3D
# ------------------------------------------------------


def cleanData(data, delimiter=' '):
    # make data to be the form that np.loadtxt can execute
    dt = []
    for i in range(len(data)):
        if data[i] == ',':
            dt.append(delimiter)
        else:
            dt.append(data[i])
    dt = [''.join(dt)]
    return dt


if __name__ == "__main__":
    # ----------------------------------read file data
    import re
    job = 'tilt0.inp'
    with open('./dataFile/' + job, 'r') as r:
        fl = r.read()
    # print(fl)

    # ------------------------------------------------------- extract the string
    xyz = re.findall(r'Node(.+?)Element, type=', fl, re.S)
    xyz = xyz[0][1:-2]

    nodes = np.loadtxt(cleanData(xyz))
    nodes = np.reshape(nodes, [int(len(nodes) / 4), 4])
    nodes = nodes[:, 1:]
    nodes = np.mat(nodes)
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
    print('ele = \n{}\n'.format(ele))
    # -----------------------------------------------------------------------------------------------
    eles = []
    for i in range(len(ele[:, 0])):
        eles.append(C3D8(number=i+1, nodes=ele[i, :].tolist()[0]))
    obj1 = Object3D(eles=eles, nodes=nodes)

    obj1.ratio_draw = 1. if nodes.max() < 50. else 1./40.
    obj1.nodes *= obj1.ratio_draw

    # ------------------------ get the surfaces and corresponding edges
    #                          surfaces: xxx.surfaces
    #                          surface -> element:   xxx.surfaces[xx]['ele'][0]
    #                          surface edges:  xxx.surfaceEdges
    obj1.getFaceEdge()

    # ----------------------------------------------------------------------------------------------- 读文件
    with open('./dataFile/trss.txt', 'r') as r:
        trss = np.loadtxt(r)
        print('trss = \n', trss)
        obj1.trss = trss
        obj1.trssMax = trss.max()
        obj1.trssMin = trss.min()

    # with open('./dataFile/gradient_.txt', 'r') as r:
    #     fl = r.read()
    #     fl = fl.split('\n')
    #     print('fl[0] = ', fl[0])
    #     print('fl[0].lstrip() = ', fl[0].lstrip())
    #     if fl[0].lstrip()[0].isalpha():
    #         del fl[0]
    #     print('fl =\n', fl)
    #     data = np.loadtxt(fl)
    #     pts = data[:, 0:2]
    #     grad = data[:, 2:]

    print('obj1._eles[1].center() =', obj1._eles[1].center())

    # -----------------------------------------------------------------------------------draw the object
    #                                                                                    with vector field
    def draw():
        coordinateLine.draw(lineWidth=3.,
                            lineLength=0.8)  # draw coordinate lines


        # --------------------------------- draw the element planes
        for face in obj1.surfaces:
            color = (obj1.trss[face['ele'][0] - 1] - obj1.trssMin) / (obj1.trssMax - obj1.trssMin)
            red, green, blue = colorBar.getColor(color)
            glColor4f(red, green, blue, 1.0)
            glMaterialfv(GL_FRONT, GL_AMBIENT, [red, green, blue])
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [red, green, blue])
            glMaterialfv(GL_FRONT, GL_SPECULAR, [red, green, blue])
            glMaterialfv(GL_FRONT, GL_EMISSION, [red, green, blue])
            glBegin(GL_POLYGON)
            for node in face['nodes']:
                glVertex3f(obj1.nodes[node-1, 0],
                           obj1.nodes[node-1, 1],
                           obj1.nodes[node-1, 2])
            glEnd()

        # ----------------------------------- draw the element edges
        glLineWidth(1.0)
        red, green, blue = 0., 0., 0.
        glColor4f(red, green, blue, 1.0)
        glMaterialfv(GL_FRONT, GL_AMBIENT, [red, green, blue])
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [red, green, blue])
        glMaterialfv(GL_FRONT, GL_SPECULAR, [red, green, blue])
        glMaterialfv(GL_FRONT, GL_EMISSION, [red, green, blue])
        glBegin(GL_LINES)
        for line in obj1.surfaceEdges:
            for node in line:
                glVertex3f(obj1.nodes[node-1, 0],
                           obj1.nodes[node-1, 1],
                           obj1.nodes[node-1, 2])
        glEnd()

        drawVectorField = False
        if drawVectorField:
            # ----------------------------------- draw the Vector Field
            red, green, blue = 1., 1., 1.
            glColor4f(red, green, blue, 1.0)
            glMaterialfv(GL_FRONT, GL_AMBIENT, [red, green, blue])
            glMaterialfv(GL_FRONT, GL_DIFFUSE, [red, green, blue])
            glMaterialfv(GL_FRONT, GL_SPECULAR, [0., 1., 1.])
            glMaterialfv(GL_FRONT, GL_EMISSION, [0., 0., 0.])
            for i in range(len(pts[:, 0])):
                r = np.array([pts[i, 0], pts[i, 1], 1.1])
                r *= obj1.ratio_draw
                r = r.tolist()
                h = (np.array(grad[i]) ** 2).sum() ** (1./2.)
                h = 0.5  # what if every gradient vector use the the same length to visualize

                # ------------------------ some special treatment for visualization
                # h = 0 if i == 1 else h
                # h = 0 if i == 21 else h
                # ------------------------ some special treatment for visualization

                h *= obj1.ratio_draw

                if obj1.ratio_draw < 0.1:
                    h *= 10.

                angle = np.degrees(np.arctan(grad[i, 1] / grad[i, 0]))

                if np.sign(grad[i, 0]) == 0:
                    angle = np.sign(grad[i, 1]) * 90.

                if grad[i, 0] < 0:
                    angle -= 180.
                angle -= 180.
                Arrow3D.show(h, 0.05, angle, r)
            # ---------------------------------------------------------------
        
        glutSwapBuffers()  # 切换缓冲区，以显示绘制内容
    
    
    mhxOpenGL.showUp(draw)

