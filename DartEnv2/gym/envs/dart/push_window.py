import OpenGL.GL as GL
import OpenGL.GLU as GLU
import OpenGL.GLUT as GLUT
import sys
import numpy as np
from pydart2.gui.opengl.scene_split import OpenGLScene_split
from pydart2.gui.glut.window import *
import os

class PushGLUTWindow(GLUTWindow):
    def __init__(self,sim,title):
        super().__init__(sim, title)
        self.scene = OpenGLScene_split(*self.window_size)
        self.folder_name = "/home/niranjan/Projects/vis_inst/pytorch-CycleGAN-and-pix2pix/datasets/push_arti_2tex/"
        for root, dirs, files in os.walk('/home/niranjan/Projects/datasets/ETH_Synthesizability/texture',
                                         topdown=False):
            pass
        self.root = root
        self.files = files
        self.filename2 = os.path.join(self.root, self.files[np.random.randint(len(self.files))])
        self.filename1 = os.path.join(self.root, self.files[np.random.randint(len(self.files))])
        self.filename3 = os.path.join(self.root, self.files[np.random.randint(len(self.files))])
        print(self.filename2,self.filename1)


    def initGL(self, w, h):
        self.scene.init()
        self.scene.set_textures(self.filename1, self.filename2, self.filename3)

    def resizeGL(self, w, h):
        self.scene.resize(w, h)


    def close(self):
        GLUT.glutDestroyWindow(self.window)
        GLUT.glutMainLoopEvent()

    def drawGL(self, ):
        self.scene.render(self.sim)
        GLUT.glutSwapBuffers()

    def runSingleStep(self):
        GLUT.glutPostRedisplay()
        GLUT.glutMainLoopEvent()

    def getFrame(self):
        self.runSingleStep()
        data = GL.glReadPixels(0, 0,
                               self.window_size[0], self.window_size[1],
                               GL.GL_RGBA,
                               GL.GL_UNSIGNED_BYTE)
        img = np.frombuffer(data, dtype=np.uint8)
        return img.reshape(self.window_size[1], self.window_size[0], 4)[::-1,:,0:3]

    def mykeyboard(self, key, x, y):
        keycode = ord(key)
        key = key.decode('utf-8')
        # print("key = [%s] = [%d]" % (key, ord(key)))

        # n = sim.num_frames()
        if keycode == 27:
            self.close()
            return
        self.keyPressed(key, x, y)

    def run(self, ):
        # Init glut
        GLUT.glutInit(())
        GLUT.glutInitDisplayMode(GLUT.GLUT_RGBA |
                                 GLUT.GLUT_DOUBLE |
                                 GLUT.GLUT_ALPHA |
                                 GLUT.GLUT_DEPTH)
        GLUT.glutInitWindowSize(*self.window_size)
        GLUT.glutInitWindowPosition(0, 0)
        self.window = GLUT.glutCreateWindow(self.title)

        GLUT.glutDisplayFunc(self.drawGL)
        GLUT.glutReshapeFunc(self.resizeGL)
        GLUT.glutKeyboardFunc(self.mykeyboard)
        GLUT.glutMouseFunc(self.mouseFunc)
        GLUT.glutMotionFunc(self.motionFunc)
        self.initGL(*self.window_size)