import pygame
import cv2
import scipy.misc

from pygame.locals import *

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

class gl_ob(object):

    def __init__(self, width, height, batch_size):
        self.width = width
        self.height = height
        self.batch_size = batch_size

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
        glPolygonMode(GL_BACK, GL_FILL)




