#coding:utf-8
from __future__ import print_function
import sys
import os
import wave
import pyaudio
import cPickle
import numpy as np
import preprocessing as pre
import feature_extraction as fe
import matplotlib.pyplot as plt
from matplotlib import cm
from subprocess import *
from PyQt4.QtGui import *
from PyQt4.QtCore import *


class Window(QMainWindow):

    def __init__(self):
        super(Window, self).__init__()
        self.name = ''
        self.model = None
        self.range = None
        self.x = 1920
        self.y = 960
        self.setGeometry(0, 0, self.x, self.y)
        self.setWindowTitle('GUI')
        self.label1 = QLabel(self)
        self.label2 = QLabel(self)
        self.label3 = QLabel(self)
        self.label4 = QLabel(self)
        self.label4.setOpenExternalLinks(True)
        self.label5 = QLabel(self)
        self.label5.setOpenExternalLinks(True)
        self.label6 = QLabel(self)
        self.label6.setOpenExternalLinks(True)
        self.label7 = QLabel(self)
        self.label7.setOpenExternalLinks(True)
        self.label8 = QLabel(self)
        self.cb1 = QComboBox(self)
        self.cb2 = QComboBox(self)
        self.bg = QPalette()
        self.pixpath = '/home/han/project/img'
        self.home()


    def home(self):
        self.bg.setBrush(QPalette.Background, QBrush(QPixmap(os.path.join(self.pixpath, 'b.png'))))
        self.setPalette(self.bg)
        label1 = QLabel(self)
        label1.setStyleSheet('background:white')
        label1.setAlignment(Qt.AlignCenter)
        label1.setText('Language:')
        label1.resize(150,30)
        label1.move(self.x/10 - 150, self.y /6-50)
        
        self.cb1.addItem('German')
        self.cb1.addItem('English')
        self.cb1.resize(150,50)
        self.cb1.move(self.x/10 -150, self.y/6)
        
        label2 = QLabel(self)
        label2.setStyleSheet('background:white')
        label2.setAlignment(Qt.AlignCenter)
        label2.setText('Input type:')
        label2.resize(150,30)
        label2.move(self.x/10+150, self.y /6-50)
        
        self.cb2.addItem('Load File')
        self.cb2.addItem('Record')
        self.cb2.resize(150,50)
        self.cb2.move(self.x/10+150, self.y/6)
        
        btn1 = QPushButton('Start', self)
        btn1.clicked.connect(self.start)
        btn1.resize(150,100)
        btn1.move(self.x/10, 2*self.y /6-50)
        

        btn2 = QPushButton('Play', self)
        btn2.clicked.connect(self.play)
        btn2.resize(150,100)
        btn2.move(self.x/10, 3*self.y /6-50)

        btn3 = QPushButton('Feature Extraction', self)
        btn3.clicked.connect(self.getFeature)
        btn3.resize(150,100)
        btn3.move(self.x/10, 4*self.y /6-50)

        btn4 = QPushButton('Classify', self)
        btn4.clicked.connect(self.classify)
        btn4.resize(150,100)
        btn4.move(self.x/10, 5*self.y /6-50)
       
    

    def start(self):
        self.label1.clear()
        self.label2.clear()
        self.label3.clear()
        self.label4.clear()
        self.label5.clear()
        self.label6.clear()
        self.label7.clear()
        self.label8.clear()
        
        if self.cb2.currentText() == 'Record':
            self.record()
        else:
            self.getFile()
    
    def getFile(self):
        dbpath = ''
        if self.cb1.currentText() == 'German':
            dbpath = '../db/ger_test/'
        else:
            dbpath = '../db/eng_test/'
        self.name = QFileDialog.getOpenFileName(self, 'Open File', dbpath, '*.wav')
        
    
    def record(self):
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000
        CHUNK = 1024
        RECORD_SECONDS = 5
        WAVE_OUTPUT_FILENAME = 'temp.wav'
         
        audio = pyaudio.PyAudio()
         
        # start Recording
        stream = audio.open(format=FORMAT, channels=CHANNELS,
                        rate=RATE, input=True,
                        frames_per_buffer=CHUNK)
        print ("recording...")
        frames = []
         
        for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
        msg = QMessageBox.information(self, 'Message', 'Finished Recording')
       
        # stop Recording
        stream.stop_stream()
        stream.close()
        audio.terminate()
         
        waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(CHANNELS)
        waveFile.setsampwidth(audio.get_sample_size(FORMAT))
        waveFile.setframerate(RATE)
        waveFile.writeframes(b''.join(frames))
        waveFile.close()
        #if self.cb1.currentText() == 'German':
        #    cmd = '{0} -v 2 "{1}" "{2}"'.format('sox', 'temp.wav', 'test.wav')
        #    call(cmd, shell=True)
        #    call(['rm', '-f', 'temp.wav'], shell=True)   
        #    self.name = 'test.wav'
        #else:
        #    cmd = '{0} -v 2 "{1}" "{2}"'.format('sox', 'temp.wav', 'test.wav')
        #    call(cmd, shell=True)
        #    call(['rm', '-f', 'temp.wav'], shell=True)   
        self.name = WAVE_OUTPUT_FILENAME
      
    def play(self):
        chunk = 1024
        wf = wave.open(str(self.name), 'rb')
        p = pyaudio.PyAudio()
        stream = p.open(format =
                        p.get_format_from_width(wf.getsampwidth()),
                        channels = wf.getnchannels(),
                        rate = wf.getframerate(),
                        output = True)

        data = wf.readframes(chunk)
        while data != '':
            stream.write(data)
            data = wf.readframes(chunk)

        stream.close()    
        p.terminate()

    def getFeature(self):
        name = str(self.name)
        feat = fe.Signal(name)
        wf = wave.open(name, 'rb')
        signal = wf.readframes(-1)
        fs = wf.getframerate()

        voiced_signal = pre.getVoicedSignal(signal, fs, 30, 3)
        signal = np.fromstring(signal, 'Int16')
        powerspec = 10*np.log10(np.square(np.absolute(np.fft.rfft(feat.frames[0]))))

        time1 = np.linspace(0, len(signal) / (fs*1.0), num=len(signal))
        time2 = np.linspace(0, len(voiced_signal) / (fs*1.0), num=len(voiced_signal))
        time3 = np.linspace(0, feat.num_frames*0.01+0.015, num=feat.num_frames)
        freq = np.linspace(0, feat.sample_rate / 2, num=len(feat.frames[0]))


        fig = plt.figure(figsize=(15,15))
        fig.subplots_adjust(right=0.9)
        sub1 = fig.add_subplot(421)
        sub1.set_title('Raw Signal')
        sub1.set_xlabel('time(sec.)')
        sub1.set_ylabel('Amplitude')
        sub1.plot(time1, signal)
        sub2 = fig.add_subplot(422)
        sub2.set_title('VAD Signal')
        sub2.set_xlabel('time(sec.)')
        sub2.set_ylabel('Amplitude')
        sub2.plot(time2, voiced_signal)
        sub3 = fig.add_subplot(423)
        sub3.set_title('Energy')
        sub3.set_xlabel('time(sec.)')
        sub3.set_ylabel('dB')
        sub3.plot(time3, feat.st_Energy())
        sub4 = fig.add_subplot(424)
        sub4.set_title('Pitch')
        sub4.set_xlabel('time(sec.)')
        sub4.set_ylabel('Semitone')
        sub4.plot(time3, feat.st_F0_Semitone())
        sub5 = fig.add_subplot(413)
        sub5.set_title('PowerSpectrum')
        sub5.magnitude_spectrum(voiced_signal, Fs=feat.sample_rate)
        sub6 = fig.add_subplot(414)
        mfcc_data= np.swapaxes(feat.mfcc(), 0 ,1)
        im1 = sub6.imshow(mfcc_data, interpolation='nearest', cmap='jet', origin='lower', aspect='auto')
        sub6.set_title('MFCC')
        sub6.set_xlabel('frame No.', labelpad=-2)
        sub6.set_ylabel('coefficients')
        cbar = fig.colorbar(im1, pad=0.2, orientation='horizontal')
        plt.tight_layout()
        plt.show()


    def classify(self):
        feat = fe.getTestFeat(str(self.name), mode=3)
        model = None
        sc = None
        currText = self.cb1.currentText()
        if currText == 'German':
            model = cPickle.load(open('berlin_final_model.pkl', 'rb'))
            sc = cPickle.load(open('berlin_final_range.pkl', 'rb'))
        else:
            model = cPickle.load(open('rav_normal_final_model.pkl', 'rb'))
            sc = cPickle.load(open('rav_normal_final_range.pkl', 'rb'))
        test_data = sc.transform(feat.reshape(1, -1))
        pred_label = int(model.predict(test_data)[0])
        # W:anger, L:boredom, E:disgust, A:fear, F:happiness, T:sadness, N:neutral
        #emotion_label = {'W': 0, 'L': 1, 'E': 2, 'A': 3, 'F': 4, 'T': 5, 'N': 6}
        self.label1.resize(300, 300)
        self.label1.move(self.x / 3 , 200)
        self.label2.setAlignment(Qt.AlignCenter)
        self.label2.resize(225, 200)
        self.label2.move(self.x / 3 , 425)
        self.label2.setFont(QFont("Roman times", 20, QFont.Bold))
        self.label2.setStyleSheet('color:white')
        self.label3.resize(600, 200)
        self.label3.move(self.x / 3-170, 645)
        self.label3.setFont(QFont("Roman times", 20, QFont.Bold))
        self.label3.setStyleSheet('color:white')
        self.label3.setAlignment(Qt.AlignCenter)
        self.label4.move(3*self.x / 5, self.y/6 + 50)
        self.label4.resize(250, 250)
        self.label5.move(3*self.x / 5 + 320, self.y/6+ 50)
        self.label5.resize(250, 250)
        self.label6.move(3*self.x / 5, self.y/6 + 390)
        self.label6.resize(250, 250)
        self.label7.move(3*self.x / 5 + 320, self.y/6 + 390)
        self.label7.resize(250, 250)
        self.label8.move(3*self.x / 5, self.y/6 - 100)
        self.label8.resize(570, 50)
        self.label8.setFont(QFont("Roman times", 20, QFont.Bold))
        self.label8.setStyleSheet('color:white')
        self.label8.setAlignment(Qt.AlignCenter)
        
        if currText == 'German':
            if pred_label == 0:
                self.label1.setPixmap(QPixmap(os.path.join(self.pixpath, 'anger.jpeg')))
                self.label2.setText('Angry')
            elif pred_label == 1:
                self.label1.setPixmap(QPixmap(os.path.join(self.pixpath, 'bored.jpeg')))
                self.label2.setText('Bored')
            elif pred_label == 2:
                self.label1.setPixmap(QPixmap(os.path.join(self.pixpath, 'disgust.jpeg')))
                self.label2.setText('Disgust')
            elif pred_label == 3:
                self.label1.setPixmap(QPixmap(os.path.join(self.pixpath, 'fear.jpeg')))
                self.label2.setText('Anxiety/Fear')
            elif pred_label == 4:
                self.label1.setPixmap(QPixmap(os.path.join(self.pixpath, 'happy.jpeg')))
                self.label2.setText('Happy')
            elif pred_label == 5:
                self.label1.setPixmap(QPixmap(os.path.join(self.pixpath, 'sad.jpeg')))
                self.label2.setText('Sad')
            elif pred_label == 6:
                self.label1.setPixmap(QPixmap(os.path.join(self.pixpath, 'neutral.jpeg')))
                self.label2.setText('Neutral')
            if pred_label in (0, 1, 2, 3, 5, 6):
                self.label3.setText('Listen to some music to be happier!')
                self.label4.setText('<a href="https://www.youtube.com/watch?v=AbPED9bisSc"><img src="../img/1d_2.jpg"></a>')
                self.label5.setText('<a href="https://www.youtube.com/watch?v=xWTiOqJqkk0"><img src="../img/mayday.jpg"></a>')    
                self.label6.setText('<a href="https://www.youtube.com/watch?v=sHD_z90ZKV0"><img src="../img/straw.jpg"></a>')    
                self.label7.setText('<a href="https://www.youtube.com/watch?v=QJO3ROT-A4E"><img src="../img/1d.jpg"></a>')
                self.label8.setText('Music Recommendation')        
    
        else:
            if pred_label in (1, 2):
                self.label1.setPixmap(QPixmap(os.path.join(self.pixpath, 'neutral.jpeg')))
                self.label2.setText('Neutral')
            elif pred_label == 3:
                self.label1.setPixmap(QPixmap(os.path.join(self.pixpath, 'happy.jpeg')))
                self.label2.setText('Happy')
            elif pred_label == 4:
                self.label1.setPixmap(QPixmap(os.path.join(self.pixpath, 'sad.jpeg')))
                self.label2.setText('Sad')
            elif pred_label == 5:
                self.label1.setPixmap(QPixmap(os.path.join(self.pixpath, 'anger.jpeg')))
                self.label2.setText('Angry')
            elif pred_label == 6:
                self.label1.setPixmap(QPixmap(os.path.join(self.pixpath, 'fear.jpeg')))
                self.label2.setText('Fear')
            elif pred_label == 7:
                self.label1.setPixmap(QPixmap(os.path.join(self.pixpath, 'disgust.jpeg')))
                self.label2.setText('Disgust')
            elif pred_label == 8:
                self.label1.setPixmap(QPixmap(os.path.join(self.pixpath, 'surprised.jpg')))
                self.label2.setText('Surprised')
            if pred_label in (1, 2, 4, 5, 6, 7):
                self.label3.setText('Listen to some music to be happier!')
                self.label4.setText('<a href="https://www.youtube.com/watch?v=AbPED9bisSc"><img src="../img/1d_2.jpg"></a>')
                self.label5.setText('<a href="https://www.youtube.com/watch?v=xWTiOqJqkk0"><img src="../img/mayday.jpg"></a>')    
                self.label6.setText('<a href="https://www.youtube.com/watch?v=sHD_z90ZKV0"><img src="../img/straw.jpg"></a>')    
                self.label7.setText('<a href="https://www.youtube.com/watch?v=QJO3ROT-A4E"><img src="../img/1d.jpg"></a>')  
                self.label8.setText('Music Recommendation')      
        
def main():
   app = QApplication(sys.argv)
   window = Window()
   window.show()
   sys.exit(app.exec_())



if __name__ == '__main__':
   main()
