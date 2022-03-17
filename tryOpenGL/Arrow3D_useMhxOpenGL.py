
import mhxOpenGL
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

if __name__ == '__main__':
    import coordinateLine
    import lightSource
    import Arrow3D


    def draw():
        coordinateLine.draw()
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

        Arrow3D.show(1., 0.2, 60., [0.2, 0.2, 0.])
        Arrow3D.show(1., 0.2, 120., [-0.2, 0.2, 0.])

        glFlush()

        # ---------------------------------------------------------------
        glutSwapBuffers()  # 切换缓冲区，以显示绘制内容

    mhxOpenGL.showUp(draw=draw)