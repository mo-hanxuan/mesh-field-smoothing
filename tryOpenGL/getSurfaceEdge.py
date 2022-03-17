# -*- coding: utf-8 -*-

# -------------------------------------------
# quidam_02.py 旋转、缩放、改变视点和参考点
# -------------------------------------------

# ------------------------------------------------------import user-define modules
import sys
sys.path.append('G:\\Python code\\Algorithms\\tryOpenGL')
import colorBar
import coordinateLine
from Element import C3D8
from Element import Object3D
# ------------------------------------------------------

def getposture():
    global EYE, LOOK_AT

    dist = np.sqrt(np.power((EYE - LOOK_AT), 2).sum())
    if dist > 0:
        phi = np.arcsin((EYE[1] - LOOK_AT[1]) / dist)
        theta = np.arcsin((EYE[0] - LOOK_AT[0]) / (dist * np.cos(phi)))
    else:
        phi = 0.0
        theta = 0.0

    return dist, phi, theta


def init():
    glClearColor(0.0, 0.0, 0.0, 1.0)  # 设置画布背景色。注意：这里必须是4个参数
    glEnable(GL_DEPTH_TEST)  # 开启深度测试，实现遮挡关系
    glDepthFunc(GL_LEQUAL)  # 设置深度测试函数（GL_LEQUAL只是选项之一）


def draw():
    global IS_PERSPECTIVE, VIEW
    global EYE, LOOK_AT, EYE_UP
    global SCALE_K
    global WIN_W, WIN_H

    # 清除屏幕及深度缓存
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

    # 设置投影（透视投影）
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()

    if WIN_W > WIN_H:
        if IS_PERSPECTIVE:
            glFrustum(VIEW[0] * WIN_W / WIN_H, VIEW[1] * WIN_W / WIN_H, VIEW[2], VIEW[3], VIEW[4], VIEW[5])
        else:
            glOrtho(VIEW[0] * WIN_W / WIN_H, VIEW[1] * WIN_W / WIN_H, VIEW[2], VIEW[3], VIEW[4], VIEW[5])
    else:
        if IS_PERSPECTIVE:
            glFrustum(VIEW[0], VIEW[1], VIEW[2] * WIN_H / WIN_W, VIEW[3] * WIN_H / WIN_W, VIEW[4], VIEW[5])
        else:
            glOrtho(VIEW[0], VIEW[1], VIEW[2] * WIN_H / WIN_W, VIEW[3] * WIN_H / WIN_W, VIEW[4], VIEW[5])

    # 设置模型视图
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()

    # 几何变换
    glScale(SCALE_K[0], SCALE_K[1], SCALE_K[2])

    # 设置视点
    gluLookAt(
        EYE[0], EYE[1], EYE[2],
        LOOK_AT[0], LOOK_AT[1], LOOK_AT[2],
        EYE_UP[0], EYE_UP[1], EYE_UP[2]
    )

    # 设置视口
    glViewport(0, 0, WIN_W, WIN_H)

    # ---------------------------------------------------------------draw the coordinates
    coordinateLine.draw()

    # ---------------------------------------------------------------
    planes = [[4, 3, 2, 1],
              [5, 6, 7, 8],
              [1, 2, 6, 5],
              [8, 7, 3, 4],
              [2, 3, 7, 6],
              [5, 8, 4, 1]]
    planes = np.mat(planes)
    planes -= 1
    planes = planes.tolist()

    edges = [[1, 2], [4, 3], [5, 6], [8, 7],
             [1, 5], [2, 6], [4, 8], [3, 7],
             [2, 3], [1, 4], [6, 7], [5, 8]]
    edges = np.mat(edges)
    edges -= 1
    edges = edges.tolist()

    # --------------------------------- draw the element planes
    for face in obj1.surfaces:
        color = float(face['ele'][0]) / len(obj1._eles)
        red, green, blue = colorBar.getColor(color)
        glColor4f(red, green, blue, 1.0)
        glMaterialfv(GL_FRONT, GL_AMBIENT, [red, green, blue])
        glMaterialfv(GL_FRONT, GL_DIFFUSE, [red, green, blue])
        glMaterialfv(GL_FRONT, GL_SPECULAR, [red, green, blue])
        glMaterialfv(GL_FRONT, GL_EMISSION, [red, green, blue])
        glBegin(GL_POLYGON)
        for node in face['nodes']:
            glVertex3f(nodes[node-1, 0],
                       nodes[node-1, 1],
                       nodes[node-1, 2])
        glEnd()

    # ----------------------------------- draw the element edges
    red, green, blue = 0., 0., 0.
    glColor4f(red, green, blue, 1.0)
    glMaterialfv(GL_FRONT, GL_AMBIENT, [red, green, blue])
    glMaterialfv(GL_FRONT, GL_DIFFUSE, [red, green, blue])
    glMaterialfv(GL_FRONT, GL_SPECULAR, [red, green, blue])
    glMaterialfv(GL_FRONT, GL_EMISSION, [red, green, blue])
    glBegin(GL_LINES)
    for line in obj1.surfaceEdges:
        for node in line:
            glVertex3f(nodes[node-1, 0],
                       nodes[node-1, 1],
                       nodes[node-1, 2])
    glEnd()
    # ---------------------------------------------------------------
    glutSwapBuffers()  # 切换缓冲区，以显示绘制内容


def reshape(width, height):
    global WIN_W, WIN_H

    WIN_W, WIN_H = width, height
    glutPostRedisplay()


def mouseclick(button, state, x, y):
    global SCALE_K
    global LEFT_IS_DOWNED
    global MOUSE_X, MOUSE_Y

    MOUSE_X, MOUSE_Y = x, y
    if button == GLUT_LEFT_BUTTON:
        LEFT_IS_DOWNED = state == GLUT_DOWN
    elif button == 3:
        SCALE_K *= 1.05
        glutPostRedisplay()
    elif button == 4:
        SCALE_K *= 0.95
        glutPostRedisplay()


def mousemotion(x, y):
    global LEFT_IS_DOWNED
    global EYE, EYE_UP
    global MOUSE_X, MOUSE_Y
    global DIST, PHI, THETA
    global WIN_W, WIN_H

    if LEFT_IS_DOWNED:
        dx = MOUSE_X - x
        dy = y - MOUSE_Y
        MOUSE_X, MOUSE_Y = x, y

        PHI += 2 * np.pi * dy / WIN_H
        PHI %= 2 * np.pi
        THETA += 2 * np.pi * dx / WIN_W
        THETA %= 2 * np.pi
        r = DIST * np.cos(PHI)

        EYE[1] = DIST * np.sin(PHI)
        EYE[0] = r * np.sin(THETA)
        EYE[2] = r * np.cos(THETA)

        if 0.5 * np.pi < PHI < 1.5 * np.pi:
            EYE_UP[1] = -1.0
        else:
            EYE_UP[1] = 1.0

        glutPostRedisplay()


def keydown(key, x, y):
    global DIST, PHI, THETA
    global EYE, LOOK_AT, EYE_UP
    global IS_PERSPECTIVE, VIEW

    if key in [b'x', b'X', b'y', b'Y', b'z', b'Z']:
        if key == b'x':  # 瞄准参考点 x 减小
            LOOK_AT[0] -= 0.01
        elif key == b'X':  # 瞄准参考 x 增大
            LOOK_AT[0] += 0.01
        elif key == b'y':  # 瞄准参考点 y 减小
            LOOK_AT[1] -= 0.01
        elif key == b'Y':  # 瞄准参考点 y 增大
            LOOK_AT[1] += 0.01
        elif key == b'z':  # 瞄准参考点 z 减小
            LOOK_AT[2] -= 0.01
        elif key == b'Z':  # 瞄准参考点 z 增大
            LOOK_AT[2] += 0.01

        DIST, PHI, THETA = getposture()
        glutPostRedisplay()
    elif key == b'\r':  # 回车键，视点前进
        EYE = LOOK_AT + (EYE - LOOK_AT) * 0.9
        DIST, PHI, THETA = getposture()
        glutPostRedisplay()
    elif key == b'\x08':  # 退格键，视点后退
        EYE = LOOK_AT + (EYE - LOOK_AT) * 1.1
        DIST, PHI, THETA = getposture()
        glutPostRedisplay()
    elif key == b' ':  # 空格键，切换投影模式
        IS_PERSPECTIVE = not IS_PERSPECTIVE
        glutPostRedisplay()

def cleanData(data, delimiter=' '):
    # make data to be the form that np.loadtxt can execute
    dt = []
    for i in range(len(data)):
        if data[i] == ',':
            dt.append(delimiter)
        elif data[i] == '\n':
            dt.append(delimiter)
        else:
            dt.append(data[i])
    dt = [''.join(dt)]
    return dt


if __name__ == "__main__":
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
    import numpy as np

    # ----------------------------------read file data
    import re
    job = 'tilt10.inp'
    with open('dataFile\\' + job, 'r') as r:
        fl = r.read()
    # print(fl)

    # ------------------------------------------------------- extract the string
    xyz = re.findall(r'Node(.+?)Element, type=', fl, re.S)
    xyz = xyz[0][1:-2]

    nodes = np.loadtxt(cleanData(xyz))
    nodes = np.reshape(nodes, [int(len(nodes) / 4), 4])
    nodes = nodes[:, 1:]
    nodes = np.mat(nodes)
    # nodes /= 40.
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
    obj1.get_nod_ele()
    # print('obj1.nod_ele =\n')
    # print(obj1.nod_ele)
    obj1.get_eleNear()
    print('obj1.eleNear =\n')
    print(obj1.eleNear)
    obj1.getFaceEdge()

    print('obj1.surfaces =', len(obj1.surfaces))
    print('obj1.faceSet =', len(obj1.faceSet))
    print('obj1.surfaceEdges =', len(obj1.surfaceEdges))

    nodes /= 40.

    # ----------------------------------------------------------------------------------------------- 写文件
    with open('dataFile\\Surfaces_' + job + '.txt', 'w') as w:
        w.write('elementNumber  node1 node2 node3 node4\n')
        for face in obj1.surfaces:
            w.write('{}  {}  {}  {}  {}\n'.format(face['ele'][0],
                                                  face['nodes'][0],
                                                  face['nodes'][1],
                                                  face['nodes'][2],
                                                  face['nodes'][3]))
    with open('dataFile\\SurfaceEdges_' + job + '.txt', 'w') as w:
        for edge in obj1.surfaceEdges:
            for nod in edge:
                w.write('{}  '.format(nod))
            w.write('\n')

    # -----------------------------------------------------------------------------------------------
    IS_PERSPECTIVE = True  # 透视投影
    print('^^^^^^^^^ 0')
    VIEW = np.array([-0.8, 0.8, -0.8, 0.8, 1.0, 20.0])  # 视景体的left/right/bottom/top/near/far六个面
    print('^^^^^^^^^ 1')
    SCALE_K = np.array([1.0, 1.0, 1.0])  # 模型缩放比例
    EYE = np.array([0.0, 0.0, 2.0])  # 眼睛的位置（默认z轴的正方向）
    print('^^^^^^^^^ 2')
    LOOK_AT = np.array([0.0, 0.0, 0.0])  # 瞄准方向的参考点（默认在坐标原点）
    EYE_UP = np.array([0.0, 1.0, 0.0])  # 定义对观察者而言的上方（默认y轴的正方向）
    WIN_W, WIN_H = 640, 480  # 保存窗口宽度和高度的变量
    LEFT_IS_DOWNED = False  # 鼠标左键被按下
    MOUSE_X, MOUSE_Y = 0, 0  # 考察鼠标位移量时保存的起始位置

    DIST, PHI, THETA = getposture()  # 眼睛与观察目标之间的距离、仰角、方位角
    # -----------------------------------------------------------------------------------------------

    glutInit()
    displayMode = GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH
    glutInitDisplayMode(displayMode)

    glutInitWindowSize(WIN_W, WIN_H)
    glutInitWindowPosition(300, 200)
    glutCreateWindow('Quidam Of OpenGL')

    init()  # 初始化画布
    glutDisplayFunc(draw)  # 注册回调函数draw()
    glutReshapeFunc(reshape)  # 注册响应窗口改变的函数reshape()
    glutMouseFunc(mouseclick)  # 注册响应鼠标点击的函数mouseclick()
    glutMotionFunc(mousemotion)  # 注册响应鼠标拖拽的函数mousemotion()
    glutKeyboardFunc(keydown)  # 注册键盘输入的函数keydown()

    glutMainLoop()  # 进入glut主循环