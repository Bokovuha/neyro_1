import os
import sys
import PyQt6
import tensorflow as tf
import numpy as np
import keras
from PyQt6.QtCore import QSize
from PyQt6.QtWidgets import QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget, QApplication, QTextEdit, \
    QHBoxLayout, QFileDialog, QMessageBox

from model import tokenizer

print("test")

switch = False

model = keras.models.load_model('C:/Users/smols/PycharmProjects/diplom/third_model')
testtextarr = []


class AnotherWindow(QWidget):

    f = open("Инструкция.txt", "r", encoding="utf-8")
    instruction = f.read()

    def __init__(self):
        super().__init__()
        layout = QVBoxLayout()
        self.textbox = QTextEdit()
        self.textbox.setReadOnly(True)
        self.textbox.setFixedSize(600, 365)
        self.setLayout(layout)
        layout.addWidget(self.textbox)

        self.textbox.setPlainText(self.instruction)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.textarr = []
        self.setFixedSize(QSize(600, 360))

        self.setWindowTitle("Анализ текста на содержание агрессивной лексики")

        self.label1 = QLabel()
        self.label1.setText("Вас приветствует программа для анализа текста на содержание агрессивной лексики")
        self.label2 = QLabel()
        self.label2.setText("Анализируемый текст")
        self.label3 = QLabel()
        self.label3.setText("Вероятность содержания агрессивной лексики")

        self.textbox = QTextEdit()
        self.textbox.setReadOnly(0)
        self.textbox.setMinimumSize(580, 50)
        self.textbox.setMaximumSize(580, 200)

        self.f_button = QPushButton("Анализировать")
        self.f_button.clicked.connect(self.the_first_button_was_clicked)

        self.s_button = QPushButton("Выбрать файл")
        self.s_button.clicked.connect(self.the_second_button_was_clicked)

        self.inf_button = QPushButton("Инструкция")
        self.inf_button.clicked.connect(self.show_new_window)

        layout = QVBoxLayout()
        sec_layout = QHBoxLayout()
        third_layout = QHBoxLayout()
        layout.addLayout(sec_layout)
        sec_layout.addWidget(self.label1)
        sec_layout.addWidget(self.inf_button)
        layout.addWidget(self.textbox)
        layout.addWidget(self.label2)
        layout.addWidget(self.label3)
        third_layout.addWidget(self.f_button)
        third_layout.addWidget(self.s_button)
        layout.addLayout(third_layout)

        container = QWidget()
        container.setLayout(layout)

        self.setCentralWidget(container)

    def show_new_window(self):
        self.win = AnotherWindow()
        self.win.setFixedSize(620,385)
        self.win.setWindowTitle("Инструкция для пользователя")
        self.win.show()

    def the_first_button_was_clicked(self):
        self.textarr.append(self.textbox.toPlainText())
        self.first_check()

    def the_second_button_was_clicked(self):
        filename = QFileDialog.getOpenFileName(self, "Выберите файл", "C://")
        if ".txt" in filename[0]:
            comments = open(filename[0], "r", encoding='utf-8')
            for line in comments:
                self.textarr.append(line)
            self.second_check()
        else:
            QMessageBox.warning(self,"Ошибка", "Выбран файл неправильного формата")

    def first_check(self):
        prediction = model.predict(np.array(tokenizer.texts_to_matrix(self.textarr)))
        count = 0
        for arr in prediction:
            for chance in arr:
                if 0.6 < chance <= 1.0:
                    self.label2.setText("Анализируемая строка:   {}".format(self.textarr[count]))
                    self.label3.setText(
                        "Скорее всего, это предложение содержит агрессивную лексику, вероятность:   {}\n".format(
                            chance))
                elif 0.0 <= chance <= 0.6:
                    self.label2.setText("Анализируемая строка:   {}".format(self.textarr[count]))
                    self.label3.setText(
                        "Скорее всего, это предложение не содержит агрессивную лексику, вероятность:   {}\n".format(
                            chance))
                else:
                    self.label2.setText("Анализируемая строка:   {}".format(self.textarr[count]))
                    self.label3.setText("Возникла ошибка при классификации\n")
                count += 1
        self.textarr.clear()

    def second_check(self):
        self.textarr[-1] = self.textarr[-1] + "\n"
        f = open("Отчёт.txt", "w")
        prediction = model.predict(np.array(tokenizer.texts_to_matrix(self.textarr)))
        count = 0
        for arr in prediction:
            for chance in arr:
                if 0.6 < chance <= 1.0:
                    f.write("Анализируемая строка:   {}".format(self.textarr[count]))
                    f.write(
                        "Скорее всего, это предложение содержит агрессивную лексику, вероятность:   {}\n\n".format(
                            chance))
                elif 0.0 <= chance <= 0.6:
                    f.write("Анализируемая строка:   {}".format(self.textarr[count]))
                    f.write(
                        "Скорее всего, это предложение не содержит агрессивную лексику, вероятность:   {}\n\n".format(
                            chance))
                else:
                    f.write("Анализируемая строка:   {}".format(self.textarr[count]))
                    f.write("Возникла ошибка при классификации\n\n")
                count += 1
        QMessageBox.information(self, "Формирование отчёта", "Был создан отчёт об анализе файла по адресу {}".format(os.path.abspath("Отчёт.txt")))
        self.textarr.clear()

app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec()