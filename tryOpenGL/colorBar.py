# -*- coding: utf-8 -*-

# -------------------------------------------
# Module
# get the color by a specific mod
# -------------------------------------------

import warnings


def getColor(x, mod=4):
    delta = 1.e-3
    if not (0. - delta <= x <= 1. + delta):
        red, green, blue = 0.2, 0.2, 0.2
        if x > 1. + delta:
            red, green, blue = 0.5, 0.5, 0.5
            warnings.warn('colorBar x > 1.')
        elif x < 0. - delta:
            red, green, blue = 0.2, 0.2, 0.2
            warnings.warn('colorBar x < 0.')
        return red, green, blue
    def case1():
        # (1.0 red -> 0.5 green -> 0.0 blue)
        if x >= 0.5:
            red = (x - 0.5) / 0.5
            green = (1. - x) / 0.5
            blue = 0.
        else:
            red = 0.
            green = x / 0.5
            blue = (0.5 - x) / 0.5
        return red, green, blue
    def case2():
        # (1, 0, 0) -> (0.5, 1, 0.5) -> (0, 0, 1)
        #   red     ->  bright green -> blue (more smooth)
        red = x
        blue = 1. - x
        green = (1. - x) / 0.5 if x >= 0.5 else x / 0.5
        return red, green, blue
    def case3():
        # (1.0 red -> 0.5 white -> 0.0 blue)
        if x >= 0.5:
            red = 1.
            green = (1. - x) / 0.5
            blue = (1. - x) / 0.5
        else:
            red = x / 0.5
            green = x / 0.5
            blue = 1.
        return red, green, blue
    def case4():
        # rainbow colorBar, 4 intervals,
        #      (1 ~  0.75  ~ 0.5   ~ 0.25 ~ 0)
        # -> (red ~ yellow ~ green ~ cyan ~ blue)
        if x >= 0.75:
            red = 1.
            green = (1. - x) / 0.25
            blue = 0.
        elif 0.5 <= x < 0.75:
            red = (x - 0.5) / 0.25
            green = 1.
            blue = 0.
        elif 0.25 <= x < 0.5:
            red = 0.
            green = 1.
            blue = (0.5 - x) / 0.25
        else:
            red = 0.
            green = x / 0.25
            blue = 1.
        return red, green, blue
    def case5():
        # (1.0 red -> 0.0 blue)  smooth across 0 -> 1
        # (1,0,0) -> (0.5,0,0.5) -> (0,0,1)
        #   red   ->    purple   ->  blue
        red = x
        green = 0.
        blue = 1. - x
        return red, green, blue
    def case6():
        # (1.0 red -> 0.5 black -> 0.0 blue)
        if x >= 0.5:
            red = (x - 1. / 2.) / (1. / 2.)
            green = 0.
            blue = 0.
        else:
            red = 0.
            green = 0.
            blue = (1. / 2. - x) / (1. / 2.)
        return red, green, blue
    def case7():
        # (1,0,0) -> (0.5,0.5,0.5) -> (0,0,1)
        #   red   ->    grey       ->  blue
        red = x
        blue = 1. - x
        green = 1. - x if x >= 0.5 else x
        return red, green, blue
    
    switch = {1:case1, 2:case2, 3:case3, 4:case4, 5:case5, 6:case6, 7:case7}
    func = switch[mod]
    return func()


if __name__ == "__main__":
    from OpenGL.GL import *
    from OpenGL.GLU import *
    from OpenGL.GLUT import *
    import numpy as np


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
        glClearColor(0.0, 0.0, 0.0, 1.0)  # ????????????????????????????????????????????????4?????????
        glEnable(GL_DEPTH_TEST)  # ???????????????????????????????????????
        glDepthFunc(GL_LEQUAL)  # ???????????????????????????GL_LEQUAL?????????????????????


    def draw():
        global IS_PERSPECTIVE, VIEW
        global EYE, LOOK_AT, EYE_UP
        global SCALE_K
        global WIN_W, WIN_H

        # ???????????????????????????
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # ??????????????????????????????
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

        # ??????????????????
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()

        # ????????????
        glScale(SCALE_K[0], SCALE_K[1], SCALE_K[2])

        # ????????????
        gluLookAt(
            EYE[0], EYE[1], EYE[2],
            LOOK_AT[0], LOOK_AT[1], LOOK_AT[2],
            EYE_UP[0], EYE_UP[1], EYE_UP[2]
        )

        # ????????????
        glViewport(0, 0, WIN_W, WIN_H)

        # --------------------------------------------------------------- draw the scaler bar

        width = 2.
        hight = 15.6

        ndy = 12
        dy = hight / ndy

        glBegin(GL_QUAD_STRIP)
        for ny in range(ndy):
            y0 = ny * dy
            r = y0 / hight
            red, green, blue = getColor(r, mod=5)
            glColor4f(red, green, blue, 1.0)
            glVertex2d(0., y0)
            glVertex2d(width, y0)

            y1 = y0 + dy
            r = y1 / hight
            red, green, blue = getColor(r, mod=5)
            glColor4f(red, green, blue, 1.0)
            glVertex2d(0., y1)
            glVertex2d(width, y1)

        glEnd()

        # ---------------------------------------------------------------
        glutSwapBuffers()  # ???????????????????????????????????????


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
            if key == b'x':  # ??????????????? x ??????
                LOOK_AT[0] -= 0.01
            elif key == b'X':  # ???????????? x ??????
                LOOK_AT[0] += 0.01
            elif key == b'y':  # ??????????????? y ??????
                LOOK_AT[1] -= 0.01
            elif key == b'Y':  # ??????????????? y ??????
                LOOK_AT[1] += 0.01
            elif key == b'z':  # ??????????????? z ??????
                LOOK_AT[2] -= 0.01
            elif key == b'Z':  # ??????????????? z ??????
                LOOK_AT[2] += 0.01

            DIST, PHI, THETA = getposture()
            glutPostRedisplay()
        elif key == b'\r':  # ????????????????????????
            EYE = LOOK_AT + (EYE - LOOK_AT) * 0.9
            DIST, PHI, THETA = getposture()
            glutPostRedisplay()
        elif key == b'\x08':  # ????????????????????????
            EYE = LOOK_AT + (EYE - LOOK_AT) * 1.1
            DIST, PHI, THETA = getposture()
            glutPostRedisplay()
        elif key == b' ':  # ??????????????????????????????
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


    # -----------------------------------------------------------------------------------------------
    IS_PERSPECTIVE = True  # ????????????
    VIEW = np.array([-0.8, 0.8, -0.8, 0.8, 1.0, 20.0])  # ????????????left/right/bottom/top/near/far?????????
    SCALE_K = np.array([1.0, 1.0, 1.0])  # ??????????????????
    EYE = np.array([0.0, 0.0, 2.0])  # ????????????????????????z??????????????????
    LOOK_AT = np.array([0.0, 0.0, 0.0])  # ???????????????????????????????????????????????????
    EYE_UP = np.array([0.0, 1.0, 0.0])  # ??????????????????????????????????????????y??????????????????
    WIN_W, WIN_H = 640, 480  # ????????????????????????????????????
    LEFT_IS_DOWNED = False  # ?????????????????????
    MOUSE_X, MOUSE_Y = 0, 0  # ?????????????????????????????????????????????

    DIST, PHI, THETA = getposture()  # ?????????????????????????????????????????????????????????
    # -----------------------------------------------------------------------------------------------

    glutInit()
    displayMode = GLUT_DOUBLE | GLUT_ALPHA | GLUT_DEPTH
    glutInitDisplayMode(displayMode)

    glutInitWindowSize(WIN_W, WIN_H)
    glutInitWindowPosition(300, 200)
    glutCreateWindow('Quidam Of OpenGL')

    init()  # ???????????????
    glutDisplayFunc(draw)  # ??????????????????draw()
    glutReshapeFunc(reshape)  # ?????????????????????????????????reshape()
    glutMouseFunc(mouseclick)  # ?????????????????????????????????mouseclick()
    glutMotionFunc(mousemotion)  # ?????????????????????????????????mousemotion()
    glutKeyboardFunc(keydown)  # ???????????????????????????keydown()

    glutMainLoop()  # ??????glut?????????