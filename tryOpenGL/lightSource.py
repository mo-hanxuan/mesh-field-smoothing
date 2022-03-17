# -*- coding: utf-8 -*-

# -------------------------------------------
# Module: setting the light source
# --------------------------------------------

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

def turnOn(position=[-80., 0., 60.],
           ambient=[0., 0., 0., 1.],
           diffuse=[1., 1., 1., 1.],
           specular=[1., 1., 1., 1.],
           light=GL_LIGHT0):  # turn on the light
    # -------------------------------------------------------------- first light source
    sun_light_position = position  # light source position
    sun_light_ambient = ambient  # RGBA模式的环境光，为0
    sun_light_diffuse = diffuse  # RGBA模式的漫反射光，全白光
    sun_light_specular = specular  # RGBA模式下的镜面光 ，全白光

    glLightfv(light, GL_POSITION, sun_light_position)
    glLightfv(light, GL_AMBIENT, sun_light_ambient)
    glLightfv(light, GL_DIFFUSE, sun_light_diffuse)
    glLightfv(light, GL_SPECULAR, sun_light_specular)

    glEnable(light)
    glEnable(GL_LIGHTING)
    glEnable(GL_DEPTH_TEST)