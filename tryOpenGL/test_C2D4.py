"""
show how to use class C2D4 in Element.py
and show how to use function c2d4.getPatch(n) to
make patches inside the quadrangle (densify the grid)
c2d4.getPatch(n), return:
    self.patches
        type: np.ndarray
        shape: (n*n, 4, 3)
            1st index enumerates each patch
            2nd index enumerates the nodes of the patch
            3rd index enumerates the global coordinates
    self.outFrame
        type: np.ndarray
        shape: (4*n, 2, 3)
            1st index enumerates each line segment
            2nd index enumerates two endpoints
            3rd index enumerates the global coordinates
"""
import numpy as np

import sys

sys.path.append('G:\\Python code\\Algorithms\\tryOpenGL')

import mhxOpenGL
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import coordinateLine
import colorBar

import Element


if __name__ == '__main__':
    x = [[-5., -5.],
         [5., -5.],
         [7., 5.],
         [-3., 5.]]

    def draw():
        coordinateLine.draw(lineWidth=3.,
                            lineLength=0.8)  # draw coordinate lines



        c2d4 = Element.C2D4(xy=x)
        c2d4.getPatch(8)  # get c2d4.patches
                          # and c2d4.outFrame

        for patch in range(len(c2d4.patches[:, 0, 0])):
            glBegin(GL_POLYGON)
            for nod in range(4):
                X = c2d4.patches[patch, nod, 0]
                Y = c2d4.patches[patch, nod, 1]
                Z = - (X / 3.)**2 - (Y / 3.)**2
                # 这个地方鲁棒性太弱了了！！！ 一定要改一下
                color = patch / len(c2d4.patches[:, 0, 0])
                red, green, blue = colorBar.getColor(color)
                glColor4f(red, green, blue, 1.0)
                glMaterialfv(GL_FRONT, GL_AMBIENT, [red, green, blue])
                glMaterialfv(GL_FRONT, GL_DIFFUSE, [red, green, blue])
                glMaterialfv(GL_FRONT, GL_SPECULAR, [red, green, blue])
                glMaterialfv(GL_FRONT, GL_EMISSION, [red, green, blue])

                glVertex3f(X,
                           Y,
                           Z)
                # print('[X, Y, Z] =', [X, Y, Z])
            glEnd()

        # # ----------------------------------- draw the element edges
        # glLineWidth(2.0)
        # red, green, blue = 0., 0., 0.
        # glColor4f(red, green, blue, 1.0)
        # glMaterialfv(GL_FRONT, GL_AMBIENT, [red, green, blue])
        # glMaterialfv(GL_FRONT, GL_DIFFUSE, [red, green, blue])
        # glMaterialfv(GL_FRONT, GL_SPECULAR, [red, green, blue])
        # glMaterialfv(GL_FRONT, GL_EMISSION, [red, green, blue])
        # glBegin(GL_LINES)
        # for ele in set1:
        #     face = ele.surfaceGlobal[0]['face']  # 这个地方鲁棒性太弱了了！！！ 一定要改一下
        #     for i in range(-1, len(face['nodes'])-2):
        #         line = [face['nodes'][i], face['nodes'][i+1]]
        #         for node in line:
        #             X = obj1.nodes[node, 0] - region_cen[0]
        #             Y = obj1.nodes[node, 1] - region_cen[1]
        #             Z = circleFit.circleFunc(w.tolist(),
        #                                      [obj1.nodes[node, 0],
        #                                       obj1.nodes[node, 1]],
        #                                      dm=2
        #                                      ).tolist()[0][0]
        #             glVertex3f(X * obj1.ratio_draw,
        #                        Y * obj1.ratio_draw,
        #                        Z * obj1.ratio_draw)
        #             # 这个地方鲁棒性太弱了了！！！ 一定要改一下
        # glEnd()
        # ---------------------------------------------------------------
        glutSwapBuffers()  # 切换缓冲区，以显示绘制内容


    mhxOpenGL.showUp(draw)
