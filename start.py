import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QPushButton, QMessageBox,QFileDialog,QLineEdit,QLabel,QComboBox
from peizhun_shipin_xiugai1 import shipin
from peizhun_shipin1 import shipin1
from PyQt5.QtGui import QGuiApplication,QCursor
from PyQt5.QtCore import Qt


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(800, 600)
        self.setWindowTitle("术中荧光分析")

        self.button1 = QPushButton('加载视频', self) # 创建一个button1
        self.button1.clicked.connect(self.open) # 将button1与self.click链接
        self.button1.setGeometry(570, 100, 93, 28)
        self.button2 = QPushButton('保存位置', self) # 创建一个button1
        self.button2.clicked.connect(self.save) # 将button1与self.click链接
        self.button2.setGeometry(570, 200, 93, 28)
        self.text1 = QLineEdit(self)
        self.text1.setReadOnly(True)
        self.text1.setGeometry(160, 100, 400, 28)
        self.text2 = QLineEdit(self)
        self.text2.setReadOnly(True)
        self.text2.setGeometry(160, 200, 400, 28)
        self.label =QLabel("尺子长度(cm):",self)
        self.label.setGeometry(160, 300, 200, 28)
        self.text3 = QLineEdit(self)
        self.text3.setGeometry(270,300, 100, 28)
        self.label1 =QLabel("荧光颜色:",self)
        self.label1.setGeometry(160, 400, 200, 28)
        self.combobox = QComboBox(self)
        self.combobox.setGeometry(270, 400, 100, 28)
        self.combobox.addItems(['绿色', '蓝色'])
        self.button3 = QPushButton('开始计算', self) # 创建一个button1
        self.button3.setGeometry(350, 500, 93, 28)
        self.button3.clicked.connect(self.start)
    def open(self):
        global fileName1
        fileName1, filetype1 = QFileDialog.getOpenFileName(self,"选取文件","./", "All Files (*);;Excel Files (*.mp4)")  #设置文件扩展名过滤,注意用双分号间隔
        print(fileName1)
        self.text1.setText(fileName1)
        if self.text1.text():
            QMessageBox.about(self, '提示', '加载成功')  # 弹窗
        else:
            QMessageBox.about(self, '提示', '加载失败，请重新加载!')  # 弹窗
    def save(self):
        global dir
        dir = QFileDialog.getExistingDirectory(self, "保存位置",
                                                "./",
                                                QFileDialog.ShowDirsOnly
                                                | QFileDialog.DontResolveSymlinks)
        print(dir)
        self.text2.setText(dir)
        if self.text2.text():
            QMessageBox.about(self, '提示', '加载成功')  # 弹窗
        else:
            QMessageBox.about(self, '提示', '加载失败，请重新加载!')  # 弹窗
    def start(self):
        QGuiApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        # savepath=dir+"/"+"qudou_"+fileName1.split('/')[-1]
        # print(savepath)
        if self.combobox.currentIndex()==0:
            print("绿色荧光")
            shipin(fileName1,dir,self.text3.text())
        else:
            print("蓝色荧光")
            shipin1(fileName1,dir,self.text3.text())
        QGuiApplication.restoreOverrideCursor()
if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())

