""" Model view controller in QT """

import sys

from PyQt4 import QtCore, QtGui

#==============================================================================
# View (only ever draws the model data and retransmits messages)
#==============================================================================

class graphicsView(QtGui.QGraphicsView):
    """
    Custom graphics view that can listen to events and pass up.
    """
    
    movement = QtCore.pyqtSignal(int, int, name='movement')
    
    def __init__(self, *args):
        QtGui.QGraphicsView.__init__(self, *args)
    
    def mouseMoveEvent(self, event):
        self.movement.emit(event.pos().x(), event.pos().y())
        
    def wheelEvent(self, event):
        print event.delta()


class dialog(QtCore.QObject):
    """
    Dialog is the "view"    """
    
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(648, 612)
        Dialog.setModal(False)
        
        self.graphicsView = graphicsView(Dialog)
        self.graphicsView.setGeometry(QtCore.QRect(10, 10, 631, 591))
        self.graphicsView.setObjectName("graphicsView")

        self.scene = QtGui.QGraphicsScene()
        self.graphicsView.setScene(self.scene)
        
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(
            QtGui.QApplication.translate(
                "Dialog",
                "View",
                None,
                QtGui.QApplication.UnicodeUTF8
                )
            )

    def draw(self, model):
        """
        Views a model as a point.
        """
        self.scene.addRect(
            QtCore.QRect(model.x, model.y, 10, 10),
            pen=QtGui.QPen()
            )


#==============================================================================
# Model
#==============================================================================

class Model():
    """
    Representation of things, like points and polygons.
    """
    def __init__(self, x, y):
        self.x = x
        self.y = y


#==============================================================================
# Controller (subscribes to messages from view)
#==============================================================================


class Controller(QtGui.QDialog):
    """ Control of view and model """

    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)

        self.view = dialog()
        self.view.setupUi(self)
        
        # Connect to messages.
        self.view.graphicsView.movement.connect(self.movement)

    # This is where the message needs to propagate to, the controller needs to
    # know that something has happened in the view.
    def movement(self, x, y):
        print 'x : {}, y : {}'.format(x, y)


if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    application = Controller()
    application.show()
    sys.exit(app.exec_())
