"""
nonlinear fitting

before fitting, the sample points are densified,
then the dense samples are taken to fitting
"""

import numpy as np
import matplotlib.pyplot as plt
import threading, time

import sys

sys.path.append('../../tryOpenGL')
sys.path.append('../')
import read_file_get_object

import mhxOpenGL
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

import coordinateLine
import colorBar
import Arrow3D

"""different nonlinear function for fitting"""
# import circleFit as nonlinear
# import neuralNetwork as nonlinear
# import ellipticFit as nonlinear
# import logisticCum as nonlinear

import Element
from read_txt import read_txt

import XYZforGL  # get the X, Y, Z of the quadrangles and lines that you need to draw 
from neuralNetwork_Fortran import NeuralNetwork # used in XYZforGL
nonlinear = NeuralNetwork()


def getData(obj1, mod='default', frame='frame33'):
    if mod == 'default':
        for ele in obj1._eles:
            cen = ele.center()
            ele.center = cen
            morph = 'ellip'
            if morph == 'bar':
                r1 = 0.5
                r2 = 0.5
                if abs(cen[0] - 0.5) < r1 and abs(cen[1] - 0.5) < r2:
                    ele.VF = 1.
                else:
                    ele.VF = 0.
            elif morph == 'ellip':
                r1 = 2.
                r2 = 2.
                if (cen[0] / r1) ** 2 + (cen[1] / r2) ** 2 < 1. ** 2:
                    ele.VF = 1.
                else:
                    ele.VF = 0.
        
        """get the Resolved Shear Stress (rss) for the object"""
        with open('data/TRSS_{}.txt'.format(job), 'r') as r:
        # with open('../../tryOpenGL/dataFile/ResolvedShearStress_' + job + '.dat', 'r') as r:
            rss = np.loadtxt(r)
            obj1.rss = rss
            obj1.rssMax = rss.max()
            obj1.rssMin = rss.min()
            # obj1.rssMax, obj1.rssMin = 810., -546.
    
    elif mod == 'read_file':
        vf = read_txt('./data/SDV210_fullField_{}.txt'.format(job))
        trss = read_txt('./data/SDV212_fullField_{}.txt'.format(job))

        for iele, ele in enumerate(obj1._eles):
            cen = ele.center()
            ele.center = cen
            ele.VF = vf[frame][iele]
        rss = trss[frame]
        obj1.rss = rss
        obj1.rssMax = rss.max()
        obj1.rssMin = rss.min()


def getSamplesBatch(
        obj1, 
        mod='trss_outlayer_dense'
    ):  
    """
        get the smaples as targets for fitting
        get different sets for vf and trss fitting
    """
    # extract a region from the object
    x1 = []
    val1 = []
    set1 = set()  # the element set of selected region
    for ele in obj1._eles:
        cen = ele.center
        if abs(cen[0] - region_cen[0]) < 5.:
            if abs(cen[1] - region_cen[1]) < 5.:
                ele.neighbor = obj1.eleNear[ele]
                set1.add(ele)
                x1.append(cen[:2])
        ele.center = cen
       
    # extract the surface region to be densified (related to vf and trss, respectively)

    if mod != 'surface_dense':
        set2_vf = set()  # vf, densified by elements whose neighbors have different vf
        for ele in set1:
            for ele2 in ele.neighbor:
                if abs(obj1._eles[ele2 - 1].VF - ele.VF) > 1.e-6:
                    set2_vf.add(ele)
        setSparse_vf = set1 - set2_vf
    else:
        set2_vf = set()
        for ele in set1:
            if (1. - ele.VF) > 1.e-6:  #  < 1
                for ele2 in ele.neighbor:
                    if abs(1. - obj1._eles[ele2 - 1].VF) < 1.e-6:  #  ≈ 1
                        set2_vf.add(ele)
        setSparse_vf = set1 - set2_vf

    #
    if mod == 'trss_outlayer_dense':
        set2_trss = set()  # trss, densified by outer layer of surface (where vf = 0)
        for ele in set1:
            if ele.VF < 1.e-6:
                for ele2 in ele.neighbor:
                    if obj1._eles[ele2 - 1].VF > 1.e-6:
                        set2_trss.add(ele)
        setSparse_trss = set1 - set2_trss
    elif mod == 'trss_outside2':
        set2_trss = set()
        for ele in set1:
            flag = 0
            if ele.VF < 1.e-6:
                for ele2_ in ele.neighbor:
                    ele2 = obj1._eles[ele2_-1]
                    if ele2.VF > 1.e-6:
                        set2_trss.add(ele)
                        flag = 1
                        break
                    print('type(ele) = {}, type(ele2) = {}'.format(type(ele), type(ele2)))
                    if hasattr(ele2, 'neighbor'):
                        print('ele2.neighbor =', ele2.neighbor)
                        for ele3_ in ele2.neighbor:
                            ele3 = obj1._eles[ele3_-1]
                            if ele3.VF > 1.e-6:
                                set2_trss.add(ele)
                                flag = 1
                                break
                    if flag == 1:
                        break
        setSparse_trss = set1 - set2_trss
    elif mod == 'trss_allOutside':
        set2_trss = set()
        setSparse_trss = set()
        for ele in set1:
            if ele.VF < 1.e-6:
                setSparse_trss.add(ele)
    elif mod == 'surface_dense':
        set2_trss = set()
        for ele in set1:
            if (1. - ele.VF) > 1.e-6:  #  < 1
                for ele2 in ele.neighbor:
                    if abs(1. - obj1._eles[ele2 - 1].VF) < 1.e-6:  #  ≈ 1
                        set2_trss.add(ele)
        setSparse_trss = set1 - set2_trss

    return \
        set1, x1, val1, \
        set2_vf, setSparse_vf, \
        set2_trss, setSparse_trss


if __name__ == '__main__':

    job = 'tilt0'
    obj1 = read_file_get_object.makeObject(job=job+'.inp', site='../../tryOpenGL/dataFile/')
    print('len(obj1._eles) =', len(obj1._eles))

    getData(obj1, mod='read_file', frame='frame21')  # tilt0
    # getData(obj1, mod='read_file', frame='frame17')  # tilt10
    # getData(obj1, mod='read_file', frame='frame34')  # tilt45

    celent = 1.

    obj1.ratio_draw = 1. if obj1.nodes.max() < 50. else 1. / 10.
    # obj1.nodes *= obj1.ratio_draw

    """center coordinates of the selected region"""
    # region_cen = [10., 2.5 * np.sqrt(3.)]
    # region_cen = [0., 5.]
    # region_cen = [20., 0.]
    # region_cen = [20., 3.]
    # region_cen = obj1._eles[6687 - 1].center()[:2]
    region_cen = [0, 0]

    print('hahaha, obj1.nodes =\n', obj1.nodes)

    
    set1, x1, val1, \
    set2_vf, setSparse_vf, \
    set2_trss, setSparse_trss \
    = getSamplesBatch(
        obj1, 
        mod='trss_allOutside'
        # mod='trss_outside',
    )


    # ================================= write the file of interface elements
    faceEle = []
    with open('data/interface_elements.txt', 'w') as file:
        file.write('%s\n' % len(set2_vf))
        for ele in set2_vf:
            file.write('%s  %s\n' % (ele.center[0], ele.center[1]))
            faceEle.append(ele.center[0:2])

    
    # ------------------------- get the [x, y] and VF that you need to fit
    x2_vf, val2_vf = [], []
    for ele in set2_vf:
        for i in [-1/3., 0, 1/3.]:
            for j in [-1/3., 0, 1/3.]:
                x = ele.center[0] + i * celent
                y = ele.center[1] + j * celent
                x2_vf.append([x, y, 0])
                if 1. - ele.VF > 1.e-6 and ele.VF > 1.e-6:
                    val2_vf.append(0.5)
                    # val2_vf.append(ele.VF)
                else:
                    val2_vf.append(ele.VF)
    for ele in setSparse_vf:  # supplyment the sparse region to set2
        x2_vf.append(ele.center)
        val2_vf.append(ele.VF)
   
    # ------------------------- get the [x, y] and trss that you need to fit
    x2_trss, val2_trss = [], []
    for ele in set2_trss:
        for i in [-1/3., 0, 1/3.]:
            for j in [-1/3., 0, 1/3.]:
                x = ele.center[0] + i * celent
                y = ele.center[1] + j * celent
                x2_trss.append([x, y, 0])
                val2_trss.append(obj1.rss[ele])
    for ele in setSparse_trss:  # supplyment the sparse region to set2
        x2_trss.append(ele.center)
        val2_trss.append(obj1.rss[ele])
    val2_trss = np.array(val2_trss)
    # val2_trss = (val2_trss - val2_trss.min()) / (val2_trss.max() - val2_trss.min())  # normalization ?
   

    print('len(set1) =', len(set1))


    with open('data/fittingData_phi.txt', 'w') as file:  # wite data file of vf
        file.write('{}\n'.format(len(val2_vf)))
        for i in range(len(val2_vf)):
            file.write('{:15.10f}  {:15.10f}  {:15.10f}\n'.format(
                val2_vf[i], x2_vf[i][0], x2_vf[i][1]
            ))
    
    with open('data/fittingData_stress.txt', 'w') as file:  # wite data file of trss
        file.write('{}\n'.format(len(val2_trss)))
        for i in range(len(val2_trss)):
            file.write('{:15.10f}  {:15.10f}  {:15.10f}\n'.format(
                val2_trss[i], x2_trss[i][0], x2_trss[i][1]
            ))

    # =================================================================================================
    # now, fit the parameters, first fit phi, than fit stress
    # =================================================================================================
    if not os.path.exists('./data'):
        os.makedirs('data')
    import platform
    sys = platform.system()
    if sys == 'Windows':
        os.system('gfortran fitting_main_getGradient.for -o fitting_main')
        os.system('fitting_main.exe')
    elif sys == 'Linux':
        os.system('gfortran fitting_main_getGradient.for -o fitting_main')
        os.system('./fitting_main')
    else:
        raise ValueError("what's your OS system, Windows or Linux ?")
    # =================================================================================================

    # ================ get parameters w from file ================
    w_phi = np.loadtxt('./data/w_phi.txt')
    print('^^^^^^^^^^^^^^^ w =\n', w_phi)
    w_stress = np.loadtxt('./data/w_freeFace.txt')
    print('^^^^^^^^^^^^^^^ w =\n', w_stress)
    w_stress_fixFace = np.loadtxt('./data/w_fixFace.txt')
    print('^^^^^^^^^^^^^^^ w =\n', w_stress_fixFace)

    lock1 = threading.Lock()

    dense_edges = True  # whether draw the dense grids inside the sparse grid


    def plt_show():

        # get the history loss of phi
        hLoss = np.loadtxt('./data/historyLoss.txt')
        if len(hLoss.shape) == 2:
            plt.figure()
            plt.plot(hLoss[:, 0], hLoss[:, 1])
            plt.title('BCE Loss of Φ vs steps', fontsize=20.)
            plt.xlabel('steps', fontsize=20.)
            plt.ylabel('loss', fontsize=20.)
            plt.xticks(fontsize=15.)
            plt.yticks(fontsize=20.)
            plt.ylim(ymin=0.)
            plt.tight_layout()
            plt.pause(1.)
        
        # get the history loss of stress
        hLoss = np.loadtxt('./data/historyLoss_freeFace.txt')
        if len(hLoss.shape) == 2:
            plt.figure()
            plt.plot(hLoss[:, 0], hLoss[:, 1]**0.5)
            plt.title('MSELoss ^ 0.5 vs steps, freeFace', fontsize=20.)
            plt.xlabel('steps', fontsize=20.)
            plt.ylabel('loss', fontsize=20.)
            plt.xticks(fontsize=15.)
            plt.yticks(fontsize=20.)
            plt.ylim(ymin=0.)
            plt.tight_layout()
            plt.pause(1.)
            print('freeFace, final loss ^ 0.5 is', hLoss[-1, 1]**0.5)

        # get the history loss
        hLoss = np.loadtxt('./data/historyLoss_fixFace.txt')
        if len(hLoss.shape) == 2:
            plt.figure()
            plt.plot(hLoss[:, 0], hLoss[:, 1]**0.5)
            plt.title('MSELoss ^ 0.5 vs steps, fixFace', fontsize=20.)
            plt.xlabel('steps', fontsize=20.)
            plt.ylabel('loss', fontsize=20.)
            plt.xticks(fontsize=15.)
            plt.yticks(fontsize=20.)
            plt.ylim(ymin=0.)
            plt.tight_layout()
            plt.pause(1.)
            print('fixFace, final loss ^ 0.5 is', hLoss[-1, 1]**0.5)

        # plot the linear classifier
        x = np.mat(x1)
        nonlinear.plot(x, w_phi)
        plt.title('linear classifier of Phase Field', fontsize=20.)
        plt.pause(1.)
        # ---------------------------------------------------------
        x = np.mat(x1)
        nonlinear.plot(x, w_stress)
        plt.title('linear classifier of Stress', fontsize=20.)
        plt.tight_layout()
        plt.pause(1.)

        tipCoord = [20.5, 0.5],  # center coordinate of this element <tilt0>
        # tipCoord = [20 + np.sqrt(2.) / 2, 0.],  # center coordinate of this element <tilt45>
        eleTip = []
        for ele in set1:
            if ((np.array(ele.center[:2]) - np.array(tipCoord))**2).sum() < 0.7**2:
                eleTip = ele._n
                print('eleTip =', eleTip)
                break


    plt_show()  # plot the history loss data and classifier lines


    def drawPhiArrow(
            w, mod='phi', 
            min0=False, 
            show_original_field=False, 
            show_arrows=True, 
        ):
        lock1.acquire()
        nonlinear = NeuralNetwork(mod=mod)
        X, Y, Z, idx, X1, Y1, Z1, \
        Xden, Yden, Zden = XYZforGL.get(
                                        obj1, w, region_cen,
                                        nonlinear,
                                        dense_region=set1,
                                        sparse_region=(set1 - set1),
                                        dense_edges=False,
                                        show_original_field=show_original_field,
                                    )
        Zmax = Z.max()
        Zmin = Z.min() if min0 == False else 0
        print('Zmin = {}, Zmax = {}'.format(Zmin, Zmax))

        grad = np.loadtxt('data/interface_gradient.txt')

        if not show_original_field:
            color = (Z - Zmin) / (Zmax - Zmin)
        else:
            color = []
            for i in range(len(Z)):
                color.append(obj1._eles[idx[i] - 1].VF)

        def draw():
            # coordinateLine.draw(lineWidth=3.,
            #                     lineLength=0.8)  # draw coordinate lines
            # --------------------------------- draw the element planes

            glBegin(GL_QUADS)
            for i in range(len(X)):
                red, green, blue = colorBar.getColor(color[i])
                glColor4f(red, green, blue, 1.0)
                glMaterialfv(GL_FRONT, GL_AMBIENT, [red, green, blue])
                glMaterialfv(GL_FRONT, GL_DIFFUSE, [red, green, blue])
                glMaterialfv(GL_FRONT, GL_SPECULAR, [red, green, blue])
                glMaterialfv(GL_FRONT, GL_EMISSION, [red, green, blue])
                glVertex3f(
                    X[i], Y[i], 0.
                )
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
                    glVertex3f(
                        Xden[i], Yden[i], 0.
                    )
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
                glVertex3f(
                    X1[i], Y1[i], 0.
                )
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

            # # ----------------------------------- draw the grad arrows (vector field)
            # if show_arrows:
            #     red, green, blue = 1., 1., 1.
            #     glColor4f(red, green, blue, 1.0)
            #     glMaterialfv(GL_FRONT, GL_AMBIENT, [red, green, blue])
            #     glMaterialfv(GL_FRONT, GL_DIFFUSE, [red, green, blue])
            #     glMaterialfv(GL_FRONT, GL_SPECULAR, [0., 1., 1.])
            #     glMaterialfv(GL_FRONT, GL_EMISSION, [0., 0., 0.])
            #     for i in range(len(grad[:, 0])):
            #         r = np.array([faceEle[i][0], faceEle[i][1], 1.1])
            #         r[0:2] -= np.array(region_cen)
            #         r *= obj1.ratio_draw
            #         r = r.tolist()
            #         h = (np.array(grad[i]) ** 2).sum() ** (1./2.)
            #         h = 1.5  # what if every grad vector use the the same length to visualize

            #         h *= obj1.ratio_draw

            #         angle = np.degrees(np.arctan(grad[i, 1] / grad[i, 0]))

            #         if np.sign(grad[i, 0]) == 0:
            #             angle = np.sign(grad[i, 1]) * 90.

            #         if grad[i, 0] < 0:
            #             angle -= 180.
            #         angle -= 180.
            #         Arrow3D.show(h, 0.05, angle, r)
            # # ---------------------------------------------------------------

            # ---------------------------------------------------------------
            glutSwapBuffers()  # 切换缓冲区，以显示绘制内容

        lock1.release()
        mhxOpenGL.showUp(draw)


    def drawOpenGL(
            w, mod='phi', 
            min0=False, 
            show_original_field=False,
        ):
        lock1.acquire()
        nonlinear = NeuralNetwork(mod=mod)
        X, Y, Z, idx, X1, Y1, Z1, \
        Xden, Yden, Zden = XYZforGL.get(
                                        obj1, w, region_cen,
                                        nonlinear,
                                        dense_region=set1,
                                        sparse_region=(set1 - set1),
                                        dense_edges=False,
                                        show_original_field=show_original_field,
                                    )
        Zmax = Z.max()
        Zmin = Z.min() if min0 == False else 0
        print('Zmin = {}, Zmax = {}'.format(Zmin, Zmax))

        if not show_original_field:
            color = (Z - Zmin) / (Zmax - Zmin)
        else:
            color = []
            for i in range(len(Z)):
                color.append(
                    (obj1.rss[idx[i] - 1] - obj1.rss.min()) / (obj1.rss.max() - obj1.rss.min())
                )


        def draw():
            # coordinateLine.draw(lineWidth=3.,
            #                     lineLength=0.8)  # draw coordinate lines
            # --------------------------------- draw the element planes

            glBegin(GL_QUADS)
            for i in range(len(X)):
                red, green, blue = colorBar.getColor(color[i])
                glColor4f(red, green, blue, 1.0)
                glMaterialfv(GL_FRONT, GL_AMBIENT, [red, green, blue])
                glMaterialfv(GL_FRONT, GL_DIFFUSE, [red, green, blue])
                glMaterialfv(GL_FRONT, GL_SPECULAR, [red, green, blue])
                glMaterialfv(GL_FRONT, GL_EMISSION, [red, green, blue])
                glVertex3f(X[i], Y[i], 0.)
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
                    glVertex3f(Xden[i], Yden[i], 0.)
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
                glVertex3f(X1[i], Y1[i], 0.)
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

        lock1.release()
        mhxOpenGL.showUp(draw)


    branch1 = threading.Thread(target=drawPhiArrow, args=(w_phi, 'phi'))
    branch1.start()
    time.sleep(8.)
    branch1_ = threading.Thread(target=drawPhiArrow, args=(w_phi, 'phi', False, True, False))
    branch1_.start()
    time.sleep(8.)
    branch2 = threading.Thread(target=drawOpenGL, args=(w_stress, 'stress'))
    branch2.start()
    time.sleep(8.)
    branch2_ = threading.Thread(target=drawOpenGL, args=(w_stress, 'stress', False, True))
    branch2_.start()
    time.sleep(8.)
    branch2_min0 = threading.Thread(target=drawOpenGL, args=(w_stress, 'stress', True))
    branch2_min0.start()
    time.sleep(8.)
    branch3 = threading.Thread(target=drawOpenGL, args=(w_stress_fixFace, 'stress'))
    branch3.start()
    time.sleep(8.)
    branch3_min0 = threading.Thread(target=drawOpenGL, args=(w_stress_fixFace, 'stress', True))
    branch3_min0.start()
    
    plt.show()

