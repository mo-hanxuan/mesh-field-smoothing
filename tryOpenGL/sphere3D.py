""" -------------------------------------------
  Module: get a sphere
  use function <show> to show the sphere
----------------------------------------------- """

import mhxOpenGL
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


def putSphere(pos, radius, color, resolution=40):  # take some parameters to show a sphere (put a sphere to the window)
    """
    the sphere is originally based at origin point (原点)
    when using glTranslatef, is the coordinate axes translate actually
    so the coordinate axes will change its position after glTranslatef !!!!!
    """
    rat = 0.5
    # ----------------- Translate and Rotate
    glTranslatef(pos[0], pos[1], pos[2])

    glutSolidSphere(GLdouble(radius), GLint(resolution), GLint(resolution))

    # ----------------- reTranslate and reRotate
    glTranslatef(-pos[0], -pos[1], -pos[2])


if __name__ == '__main__':
    # ----------------------- self-defined module
    import coordinateLine
    import lightSource
    # -----------------------

    def draw():
        coordinateLine.draw()  # draw the coordinates

        # --------------------------------------------------------------- set the light source
        lightSource.turnOn(position=[-80., 0., 60.], light=GL_LIGHT0)
        lightSource.turnOn(position=[80., 0., 60.], light=GL_LIGHT1)
        # -------------------------------------------------------------- set material parameters
        mat_ambient = [1., 1., 1., 1.]  # 定义材质的环境光颜色，白色
        mat_diffuse = [0., 0., 1., 1.]  # 定义材质的漫反射光颜色，偏蓝色
        mat_specular = [1., 0., 0., 1.]  # 定义材质的镜面反射光颜色，红色
        mat_emission = [0., 0., 0., 1.]  # 定义材质的辐射光颜色，为0
        mat_shininess = 30.0

        glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse)
        glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular)
        glMaterialfv(GL_FRONT, GL_EMISSION, mat_emission)
        glMaterialf(GL_FRONT, GL_SHININESS, mat_shininess)

        putSphere([0, 0, 0], 2., [1, 1, 1])
        putSphere([4., 0, 0], 2., [1, 1, 1])
        putSphere([0., 4, 0], 2., [1, 1, 1])

        glFlush()

        # ---------------------------------------------------------------
        glutSwapBuffers()  # 切换缓冲区，以显示绘制内容

    mhxOpenGL.showUp(draw)
