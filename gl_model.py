import pygame
import cv2
import scipy.misc

from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *

import numpy as np

class gl_ob(object):

    def __init__(self, width, height, batch_size, mode = 'random'):
        self.width = width
        self.height = height
        self.batch_size = batch_size
        self.mode = mode

        self.verticies = (
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

        self.edges = (
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

        self.surfaces = (
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

    def cube(self):

        glLineWidth(3)
        glBegin(GL_QUADS)
        for surface in self.surfaces:
            x = 0
            for vertex in surface:
                x += 1
                # glColor3fv(colors[x])
                glColor3fv((0, 0, 0))
                glVertex3fv(self.verticies[vertex])
        glEnd()

        glLineWidth(3)
        glBegin(GL_LINES)
        glColor3fv((1, 1, 1))
        for edge in self.edges:
            for vertex in edge:
                glVertex3fv(self.verticies[vertex])
        glVertex3fv((-1, .01, .01))
        glVertex3fv((1, .01, .01))
        glEnd()


    def draw_ob(self):

        pygame.init()
        display = (self.height, self.height)
        window = pygame.display.set_mode(display, DOUBLEBUF | OPENGL)

        gluPerspective(45, (float(display[0]) / float(display[1])), 0.1, 50.0)

        glTranslatef(0.0, 0.0, -5)
        glClearDepth(1.0)
        glDepthFunc(GL_LESS)
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_POINT_SMOOTH)
        glPolygonMode(GL_FRONT, GL_FILL)
        glPolygonMode(GL_BACK, GL_FILL)\

        batch_example = np.zeros((self.batch_size, self.width, self.height))
        batch_y = np.zeros(self.batch_size)
        start_y = 0

        index = 0
        x_aix = np.random.uniform(-1, 1)
        y_aix = np.random.uniform(-1, 1)
        z_aix = np.random.uniform(-1, 1)

        while True:

            # if index == 0:
            #     glRotatef(np.random.uniform(-180, 180), np.random.uniform(-1, 1), np.random.uniform(-1, 1),
            #               np.random.uniform(-1, 1))
            # else:
            #     glRotatef(np.random.random(), x_aix, y_aix, z_aix)
            if self.mode == 'random':

                glRotatef(np.random.uniform(-180, 180), np.random.uniform(-1, 1), np.random.uniform(-1, 1),
                          np.random.uniform(-1, 1))
            else:
                if index == 0:
                    start_y = np.random.uniform(-180, 180)
                    batch_y[index] = start_y
                    glRotatef(start_y, 0, 1, 0)
                else:
                    add_y = np.random.uniform(-180,180)
                    start_y = start_y + add_y
                    batch_y[index] = start_y
                    glRotatef(add_y, 0, 1, 0)

            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
            self.cube()

            string_image = pygame.image.tostring(window, 'RGB')
            temp_surf = pygame.image.fromstring(string_image, display, 'RGB')
            tmp_arr = pygame.surfarray.array3d(temp_surf)
            tmp_arr_gray = cv2.cvtColor(tmp_arr, cv2.COLOR_RGB2GRAY).T
            # for i in range(self.width):
            #     for j in range(self.height):
                    # if tmp_arr_gray[i,j]>0:
                    #     tmp_arr_gray[i,j] = 255
                    # if tmp_arr_gray[i,j] == 0:
                    #     print('HIHIHIHIHI')
                    # elif tmp_arr_gray[i,j] == 255:
                    #     print('KKKKKKKKK')
            batch_example[index,:,:] = tmp_arr_gray

            pygame.display.flip()
            pygame.time.wait(0)
            index = index + 1
            if index >= self.batch_size:
                pygame.quit()
                return batch_example, batch_y



    def show_example(self):

        example, example_y = self.draw_ob()
        # print example_y
        # print np.shape(example)

        for i in range(self.batch_size):
            im = example[i]
            scipy.misc.imsave('./sample/{}.jpg'.format(i), im)