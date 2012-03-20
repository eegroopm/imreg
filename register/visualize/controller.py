""" Model view controller in QT """

import sys

from PyQt4 import QtCore, QtGui

import shapely.geometry as geometry

#==============================================================================
# View (only ever draws the model data and retransmits messages)
#==============================================================================

class graphicsView(QtGui.QGraphicsView):
    """
    Custom graphics view that can listen to events and pass up.
    """
    
    movement = QtCore.pyqtSignal(int, int, name='movement')
    click = QtCore.pyqtSignal(int, int, name='click')
    shiftclick = QtCore.pyqtSignal(int, int, name='shiftclick')
    
    def __init__(self, *args):
        QtGui.QGraphicsView.__init__(self, *args)
        
        self.setMouseTracking(True)
    
    def mouseMoveEvent(self, event):
        point = self.mapToScene(event.pos())
        self.movement.emit(point.x(), point.y())
    
    def mousePressEvent(self, event):
        point = self.mapToScene(event.pos())
        
        if int(event.modifiers()) == QtCore.Qt.ShiftModifier:
            self.shiftclick.emit(point.x(), point.y())
        else:
            self.click.emit(point.x(), point.y())
                
    def wheelEvent(self, event):
        factor = 1.41 ** (event.delta() / 240.0)
        self.scale(factor, factor)
        

class dialog(QtCore.QObject):
    """
    Dialog is the "view"    
    """
    
    selected = None
    
    pointUpdate = QtCore.pyqtSignal(int, int, geometry.Point, name='pointupdate')
    
    
    def setupUi(self, Dialog, models):
        Dialog.setObjectName("Dialog")
        Dialog.resize(648, 612)
        Dialog.setModal(False)
        
        self.models = models
        self.graphicMap = {}
        self.modelMap = {}
        
        self.graphicsView = graphicsView(Dialog)
        self.graphicsView.setGeometry(QtCore.QRect(10, 10, 631, 591))
        self.graphicsView.setObjectName("graphicsView")

        self.scene = QtGui.QGraphicsScene()
        self.graphicsView.setScene(self.scene)
        
        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        
        self.scene.addPixmap(
            QtGui.QPixmap("africa-map1.jpg")
            )
        
    def retranslateUi(self, Dialog):
        Dialog.setWindowTitle(
            QtGui.QApplication.translate(
                "Dialog",
                "View",
                None,
                QtGui.QApplication.UnicodeUTF8
                )
            )
    
        
    def update(self):
        """
        Update objects
        """

        # Populate graphic map.
        for entry in self.models:

            if entry in self.graphicMap:
                self.graphicMap[entry].setPen(QtGui.QPen(QtGui.QColor("red")))
                continue

            elipse = QtGui.QGraphicsEllipseItem(entry.x, entry.y, 5, 5)
            elipse.setPen(QtGui.QPen(QtGui.QColor("red")))
            elipse.setBrush(
                QtGui.QBrush(
                    QtGui.QColor(128, 128, 128, alpha=128),
                    QtCore.Qt.SolidPattern
                    )
                )
            elipse.setToolTip(
                QtCore.QString('({},{})'.format(entry.x, entry.y))
                )
            self.scene.addItem(elipse)

            self.graphicMap[entry] = elipse
            self.modelMap[elipse] = entry

        if self.selected in self.modelMap:
            model = self.modelMap[self.selected]
            self.pointUpdate.emit(10, 10, model)

        self.selected = None
        
    def select(self, (x, y)):
        """
        Views a model as a point.
        """
        try:
            item = self.scene.itemAt(x, y)
            item.setPen(QtGui.QPen(QtGui.QColor("yellow")))
            self.selected = item
        except AttributeError as error:
            pass


    def move(self, (x, y)):
        if self.selected:
            self.selected.setRect(x, y, 5, 5)


#==============================================================================
# Model
#==============================================================================

#point = geometry.Point()
#ine = geometry.linestring()


#==============================================================================
# Controller (subscribes to messages from view)
#==============================================================================


class Controller(QtGui.QDialog):
    """ Control of view and model """

    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)

        self.models = []

        self.view = dialog()
        self.view.setupUi(self, self.models)
        
        # Connect to messages.
        self.view.graphicsView.movement.connect(self.movement)
        self.view.graphicsView.click.connect(self.click)
        self.view.graphicsView.shiftclick.connect(self.shiftclick)
        self.view.pointUpdate.connect(self.updatePoint)
        
    # This is where the message needs to propagate to, the controller needs to
    # know that something has happened in the view.
    def movement(self, x, y):
        # Tell the view to draw.
        self.view.move((x, y))

    def click(self, x, y):
        # Append a new point "model". 
        self.models.append(
            geometry.Point(x, y)
            )
        self.view.update()

    def shiftclick(self, x, y):
        self.view.select((x,y))
    
    def updatePoint(self, x, y, point):
        point.coords = (x, y)
    
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    application = Controller()
    application.show()
    sys.exit(app.exec_())
