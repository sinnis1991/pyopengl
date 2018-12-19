

import pygame
import cv2
import scipy.misc
import time
# from OpenGL import glEnalbe
from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

# from StringIO import StringIO
# from PIL import Image

import math
import numpy as np

# global speed, x, y, z
speed = 0.5
x = np.random.uniform(-1,1)
y = np.random.uniform(-1.1)
z = np.random.uniform(-1,1)

verticies = (
    (-1, -1, 1), #0
    (1, -1, 1),  #1
    (1, 0, 1),   #2
    (-1, 0, 1),  #3
    (-1, 0, 0),  #4
    (1, 0, 0),   #5
    (1, 1, 0),   #6
    (-1, 1, 0),  #7
    (-1, 1, -1), #8
    (-1, -1, -1),#9
    (1, -1, -1), #10
    (1, 1, -1),  #11
    )

edges = (
    (0,1), #0
    (1,2), #1
    (2,3), #2
    (3,0), #3
    (2,5), #4
    (4,5), #5
    (3,4), #6
    (5,6), #7
    (6,7), #8
    (4,7), #9
    (7,8), #10
    (8,11),#11
    (6,11),#12
    (10,11),#13
    (1,10),#14
    (9,10),#15
    (8,9), #16
    (0,9)  #17
    )

colors = (
    (1,0,0),
    (0,1,0),
    (0,0,1),
    (1,1,0),
    (0,1,1),
    (1,0,1),
    (1,0,0),
    (0,1,0),
    (0,0,1),
    (1,0,0),
    (0,1,0),
    (0,0,1)
    )

surfaces = (
    (0,3,2,1),
    (2,3,4,5),
    (5,4,7,6),
    (6,7,8,11),
    (10,9,8,11),
    (10,9,0,1),

    (1,2,5,10),
    (5,6,11,10),
    (0,9,4,3),
    (4,9,8,7)


    )

def Cube():

    glRotatef(speed, 1, 0, 0)

    glLineWidth(3)
    glBegin(GL_QUADS)
    for surface in surfaces:
        x = 0
        for vertex in surface:
            x+=1
            # glColor3fv(colors[x])
            glColor3fv((0,0,0))
            glVertex3fv(verticies[vertex])
    glEnd()

    glLineWidth(3)
    glBegin(GL_LINES)
    glColor3fv((1,1,1))
    for edge in edges:
        for vertex in edge:      
            glVertex3fv(verticies[vertex])
    glVertex3fv((-1,.01,.01))
    glVertex3fv(( 1,.01,.01))          

    glEnd()

    # glPointSize(2)
    # glBegin(GL_POINTS)
    # for vertex in verticies:
    #     glColor3fv((1,1,1))
    #     glVertex3fv(vertex)
    # glEnd()    






def main():
    pygame.init()
    display = (256,256)
    window = pygame.display.set_mode(display, DOUBLEBUF|OPENGL)

    gluPerspective(45, (float(display[0])/float(display[1])), 0.1, 50.0)

    glTranslatef(0.0,0.0, -5)
    # glClearDepth(1.0) 
    # glDepthFunc(GL_LESS)            
    glEnable(GL_DEPTH_TEST)
    # glEnable(GL_CULL_FACE)
    # glCullFace(GL_BACK)
    glEnable(GL_POINT_SMOOTH)
    glPolygonMode(GL_FRONT, GL_FILL)
    glPolygonMode(GL_BACK, GL_FILL)
    # glRotatef(1, 1, 1, 0)
    # glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    # Cube()

    batch_size = 64
    batch_example = np.zeros((64,256,256))
    index = 0

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
        # global speed, x, y, z
        if index == 0:
            glRotatef(np.random.uniform(-180,180), np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-1,1))
        else:
            glRotatef(np.random.random(), 1, 0, 0)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
        Cube()


        # x = x + np.random.random()-0.5
        # y = y + np.random.random()-0.5
        # z = z + np.random.random()-0.5

        # frame = QueryFrame(capture)                    # get a video frame using OpenCV

        # pygame.image.save(window, "./screenshot{}.jpeg".format(index))
        # data = pygame.image.tostring(window, 'RGB')
        # img = Image.frombytes('RGB', display, data)
        # print size(img)
        # window.lock()
        # data = pygame.surfarray.array3d(window)
        # pygame.pixelcopy.surface_to_array(data,window)
        # print type(window)
        # imgdata = pygame.surfarray.array2d(window)
        string_image = pygame.image.tostring(window, 'RGB')
        temp_surf = pygame.image.fromstring(string_image,display,'RGB' )
        tmp_arr = pygame.surfarray.array3d(temp_surf)
        tmp_arr2 = cv2.cvtColor(tmp_arr,cv2.COLOR_RGB2GRAY).T
        # print np.shape(tmp_arr2)


        # window.unlock()
        # print img
        # scipy.misc.imsave('./memory_sample/{}.jpg'.format(index), tmp_arr2) 
        # img = rgb2gray(img)

        # print np.shape(img)

        # batch_example[index] = img

        pygame.display.flip()
        pygame.time.wait(10)
        index = index+1
        # if index >= batch_size:
        #     pygame.quit()
        #     quit()









main()