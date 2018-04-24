import OpenGL.GL as GL
import OpenGL.GLU as GLU
import OpenGL.GLUT as GLUT
import sys
import numpy as np
from pydart2.gui.opengl.scene import OpenGLScene
from pydart2.gui.glut.window import *


class StaticGLUTWindow(GLUTWindow):
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