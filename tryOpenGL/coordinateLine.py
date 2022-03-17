# ------------------------------------------------
# draw the line of coordinate
# ------------------------------------------------

from OpenGL.GL import *


def draw(lineWidth = 3.0,
         lineLength = 0.8):
    # ---------------------------------------------------------------
    glLineWidth(lineWidth)
    glBegin(GL_LINES)  # 开始绘制线段（世界坐标系）

    # 以红色绘制x轴
    glColor4f(1.0, 0.0, 0.0, 1.0)  # 设置当前颜色为红色不透明
    glMaterialfv(GL_FRONT, GL_AMBIENT, [1., 0., 0.])
    glMaterialfv(GL_FRONT, GL_DIFFUSE, [1., 0., 0.])
    glMaterialfv(GL_FRONT, GL_SPECULAR, [1., 0., 0.])
    glMaterialfv(GL_FRONT, GL_EMISSION, [1., 0., 0.])
    glVertex3f(-lineLength, 0.0, 0.0)  # 设置x轴顶点（x轴负方向）
    glVertex3f(lineLength, 0.0, 0.0)  # 设置x轴顶点（x轴正方向）

    # 以绿色绘制y轴
    glColor4f(0.0, 1.0, 0.0, 1.0)  # 设置当前颜色为绿色不透明
    glMaterialfv(GL_FRONT, GL_AMBIENT, [0., 1., 0.])
    glMaterialfv(GL_FRONT, GL_DIFFUSE, [0., 1., 0.])
    glMaterialfv(GL_FRONT, GL_SPECULAR, [0., 1., 0.])
    glMaterialfv(GL_FRONT, GL_EMISSION, [0., 1., 0.])
    glVertex3f(0.0, -lineLength, 0.0)  # 设置y轴顶点（y轴负方向）
    glVertex3f(0.0, lineLength, 0.0)  # 设置y轴顶点（y轴正方向）

    # 以蓝色绘制z轴
    glColor4f(0.0, 0.0, 1.0, 1.0)  # 设置当前颜色为蓝色不透明
    glMaterialfv(GL_FRONT, GL_AMBIENT, [0., 0., 1.])
    glMaterialfv(GL_FRONT, GL_DIFFUSE, [0., 0., 1.])
    glMaterialfv(GL_FRONT, GL_SPECULAR, [0., 0., 1.])
    glMaterialfv(GL_FRONT, GL_EMISSION, [0., 0., 1.])
    glVertex3f(0.0, 0.0, -lineLength)  # 设置z轴顶点（z轴负方向）
    glVertex3f(0.0, 0.0, lineLength)  # 设置z轴顶点（z轴正方向）

    glEnd()  # 结束绘制线段
    glLineWidth(1.0)  # reset default value