""" -------------------------------------------
  Module: get an arrow
  use function <show> to show the arrow
----------------------------------------------- """

import mhxOpenGL
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


def show(h, w_h, angle, r):  # take some parameters to show an arrow
    """
    the angle means tilt from x axes counter-clockwise,
    rotate around z axes

    the cone is originally based on z axes
    when using glRotatef, is the coordinate axes rotating actually
    so the coordinate axes will change its direction after glRotatef !!!!!
    """
    w = h * w_h  # width of cone base
    rat = 0.5
    # ----------------- Translate and Rotate
    glTranslatef(r[0], r[1], r[2])
    glRotatef(90, 0, 1, 0)
    glRotatef(-angle, 1, 0, 0)

    glutSolidCylinder(w / 2., h * rat, GLint(20), GLint(20))
    glTranslatef(0., 0., h * rat)
    glutSolidCone(w, h * (1. - rat), GLint(20), GLint(20))
    glTranslatef(0., 0., -h * rat)

    # ----------------- reTranslate and reRotate
    glRotatef(angle, 1, 0, 0)
    glRotatef(-90, 0, 1, 0)
    glTranslatef(-r[0], -r[1], -r[2])


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
        mat_ambient = [1., 1., 1., 1.]  # 定义材质的环境光颜色，蓝色
        mat_diffuse = [0., 0., 1., 1.]  # 定义材质的漫反射光颜色，偏蓝色
        mat_specular = [1., 0., 0., 1.]  # 定义材质的镜面反射光颜色，红色
        mat_emission = [0., 0., 0., 1.]  # 定义材质的辐射光颜色，为0
        mat_shininess = 30.0

        glMaterialfv(GL_FRONT, GL_AMBIENT, mat_ambient)
        glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_diffuse)
        glMaterialfv(GL_FRONT, GL_SPECULAR, mat_specular)
        glMaterialfv(GL_FRONT, GL_EMISSION, mat_emission)
        glMaterialf(GL_FRONT, GL_SHININESS, mat_shininess)

        show(1., 0.2, 60., [0.2, 0.2, 0.])
        show(1., 0.2, 120., [-0.2, 0.2, 0.])

        glFlush()

        # ---------------------------------------------------------------
        glutSwapBuffers()  # 切换缓冲区，以显示绘制内容

    mhxOpenGL.showUp(draw)
