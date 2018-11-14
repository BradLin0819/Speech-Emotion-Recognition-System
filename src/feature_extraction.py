from __future__ import print_function
import sys
import math
import preprocessing as pre
import numpy as np
from scipy.fftpack import dct
from scipy.signal import lfilter
from scipy.signal import argrelextrema
from scikits.talkbox import lpc
import matplotlib.pyplot as plt


class Signal:


    def __init__(self, path, win_ms=0.025, overlap=0.015, mode=2, preemp=0.97, nfft=512):

        self.audio, self.sample_rate = pre.read_data(path)
        self.voiced_signal = pre.getVoicedSignal(self.audio, self.sample_rate, mode = mode)
        self.frames = pre.framing(self.voiced_signal, self.sample_rate, win_ms, overlap, preemp)
        self.num_frames = len(self.frames)
        self.pspec = pre.powspec(self.frames, nfft)


    def st_Energy(self):
        """ 
        parameters:
           frames: frames of pre_processed signal
        return:
           ndarray, energy of frames
        """
        try:
            energy = np.zeros((self.num_frames, 1))
            for i in range(self.num_frames):
                frame = self.frames[i]
                frame = np.where(frame == 0, np.finfo(float).eps, frame)
                energy[i] = 10*np.log10(sum(frame**2))

            return energy

        except:
            raise Exception("st_Energy error")


    def st_Zcr(self):
        """ 
        parameters:
           frames: frames of pre_processed signal
        return:
           ndarray, zcr of frames
        """
        try:
            zcr = np.zeros((self.num_frames, 1))
            for i in range(self.num_frames):
                frame = self.frames[i]
                frame = frame - np.mean(frame)
                zcr[i] = np.sum(np.abs(np.sign(np.diff(frame)))) / 2 / len(frame)
            return zcr
  
        except:
          raise Exception("st_Zcr error")
        

    def st_F0_Semitone(self):
    #TODO
        semitone = []
        for frame in self.frames:
            length = len(frame / 2)
            #amdf = np.zeros(length)
            #for i in range(length):
            #    amdf[i] = np.sum(frame[i:length] - frame[0:length-i]) / (length-i) * 1.0
            #acf[0:int(self.sample_rate / 1000)] = -acf[0]
            #theta = .4 * (np.max(amdf) + np.min(amdf))
            #amdf = np.where(amdf < theta, 1, 0)
            acf = np.zeros(length)
            for i in range(length):
                acf[i] = np.sum(frame[i:length] * frame[0:length-i])


            acf[0:int(self.sample_rate / 1000)] = -acf[0]
            pp = np.argmax(acf)
            f0 = self.sample_rate * 1.0 / pp
            semitone.append(hz2semitone(f0))

        return np.asarray(semitone)
            
    
    def formant(self):
        formants = []
        for ps in self.pspec:

            #f1 = np.argmax(ps) * (self.sample_rate / 2 / ((len(ps) - 1) * 1.0))
            #temp.append(f1)
            #f2 = np.argpartition(ps, -2)[-2] * (self.sample_rate / 2 / ((len(ps) - 1) * 1.0))
            #temp.append(f2)
            freq = argrelextrema(ps, np.greater)
            f1 = freq[0][1] * self.sample_rate * 1.0 / len(ps)
            formants.append(f1)
                
        return np.asarray(formants)    
  
    def mfcc(self,numcep=13, nfilt=26, nfft=512, lowfreq=0, highfreq=None, ceplifter=22, appendEnergy=True):
        feat, energy = self.logfbank(nfilt, nfft, lowfreq, highfreq)
        feat = dct(feat, type=2, axis=1, norm='ortho')[:,:numcep]
        feat = self.lifter(feat,ceplifter)
        if appendEnergy: feat[:,0] = np.log(energy) # replace first cepstral coefficient with log of frame energy
        return feat

    
    def medc(self,nfilt=12, nfft=512, lowfreq=0, highfreq=None):
        feat, energy = self.logfbank(nfilt)
        feat /= (nfilt*1.0)
        feat_1st = np.diff(feat) #first difference
        feat_2nd = np.diff(feat_1st) #second difference
        feat = np.hstack((feat_1st,feat_2nd))
      
        return feat


    def fbank(self, nfilt=26, nfft=512, lowfreq=0, highfreq=None):
        highfreq = highfreq or self.sample_rate / 2
        pspec = self.pspec
        energy = np.sum(pspec, 1) # this stores the total energy in each frame
        energy = np.where(energy == 0, np.finfo(float).eps,energy) # if energy is zero, we get problems with log
        fb = get_filterbanks(nfilt, nfft, self.sample_rate, lowfreq, highfreq)
        feat = np.dot(pspec, fb.T) # compute the filterbank energies
        feat = np.where(feat == 0,np.finfo(float).eps,feat) # if feat is zero, we get problems with log

        return feat, energy
   

    def logfbank(self, nfilt=26, nfft=512, lowfreq=0, highfreq=None):
        feat, energy = self.fbank(nfilt,nfft,lowfreq,highfreq)

        return np.log(feat), energy
   

    def lifter(self, cepstra, L=22):
        """Apply a cepstral lifter the the matrix of cepstra. This has the effect of increasing the
        magnitude of the high frequency DCT coeffs.
        :param cepstra: the matrix of mel-cepstra, will be numframes * numcep in size.
        :param L: the liftering coefficient to use. Default is 22. L <= 0 disables lifter.
        """
        if L > 0:
            nframes,ncoeff = np.shape(cepstra)
            n = np.arange(ncoeff)
            lift = 1 + (L/2.)*np.sin(np.pi*n/L)
            return lift*cepstra
        else:
            # values of L <= 0, do nothing
            return cepstra


    def delta(self, feat, N):
        """Compute delta features from a feature vector sequence.
        :param feat: A numpy array of size (NUMFRAMES by number of features) containing features. Each row holds 1 feature vector.
        :param N: For each frame, calculate delta features based on preceding and following N frames
        :returns: A numpy array of size (NUMFRAMES by number of features) containing delta features. Each row holds 1 delta feature vector.
        """
        if N < 1:
            raise ValueError('N must be an integer >= 1')
        NUMFRAMES = len(feat)
        denominator = 2 * sum([i**2 for i in range(1, N+1)])
        delta_feat = np.empty_like(feat)
        padded = np.pad(feat, ((N, N), (0, 0)), mode='edge')   # padded version of feat
        for t in range(NUMFRAMES):
            delta_feat[t] = np.dot(np.arange(-N, N+1), padded[t : t+2*N+1]) / denominator   # [t : t+2*N+1] == [(N+t)-N : (N+t)+N+1]
        
        return delta_feat
   
   
    def pickFeatures(self, pick = (0, 0, 0, 1, 1)):
        """
        max,min,amean,stddev
        """
        features = []

        if pick[0] == 1:
            energy = self.st_Energy()
            features.append(energy.max())
            features.append(energy.min())
            features.append(energy.mean())
            features.append(energy.std())

        if pick[1] == 1:
            pitch = self.st_F0_Semitone()
            features.append(pitch.max())
            features.append(pitch.min())
            features.append(pitch.mean())
            features.append(pitch.std())
       
        if pick[2] == 1:
            formants = self.formant()
            features.append(formants.max())
            features.append(formants.min())
            features.append(formants.mean())
            features.append(formants.std())

        if pick[3] == 1:
            mfcc = self.mfcc()
            mfcc_de = self.delta(mfcc, 2)
            #mfcc_de_de = self.delta(mfcc_de, 2)
            for i in range(mfcc.shape[1]):
                features.append(mfcc[:,i].max())
                features.append(mfcc[:,i].min())
                features.append(mfcc[:,i].mean())
                features.append(mfcc[:,i].std())

            for i in range(mfcc_de.shape[1]):
                features.append(mfcc_de[:,i].max())
                features.append(mfcc_de[:,i].min())
                features.append(mfcc_de[:,i].mean())
                features.append(mfcc_de[:,i].std())
            """
            for i in range(mfcc_de_de.shape[1]):
                features.append(mfcc_de_de[:,i].max())
                features.append(mfcc_de_de[:,i].min())
                features.append(mfcc_de_de[:,i].mean())
                features.append(mfcc_de_de[:,i].std())
            """
      
        if pick[4] == 1:
            medc = self.medc()
            #TODO
            
            for i in range(medc.shape[1]):
                features.append(medc[:,i].max())
                features.append(medc[:,i].min())
                features.append(medc[:,i].mean())
                features.append(medc[:,i].std())
     
        features = np.asarray(features)
      
        return features
   

def allLabels(length = (1, 1, 1, 13, 21), pick = (0, 0, 0, 1, 1)):
      
    labels = []

    if pick[0] == 1:
        for i in range(length[0]):
            for j in ('_max', '_min', '_mean', '_std'):
                labels.append('energy'+j)
    if pick[1] == 1:
        for i in range(length[1]):
            for j in ('_max', '_min', '_mean', '_std'):
                labels.append('pitch'+j)
    if pick[2] == 1:
        for i in range(length[2]):
            for j in ('_max', '_min', '_mean', '_std'):
                labels.append('F'+str(i+1)+j)
    if pick[3] == 1:
        for i in range(length[3]):
            for j in ('_max', '_min', '_mean', '_std'):
                labels.append('mfcc'+'['+str(i)+']'+j)
        for i in range(length[3]):
            for j in ('_max', '_min', '_mean', '_std'):
                labels.append('mfcc_de'+'['+str(i)+']'+j)
    if pick[4] ==1:
        for i in range(length[4]):
            for j in ('_max', '_min', '_mean', '_std'):
                labels.append('medc'+'['+str(i)+']'+j)
   
    return labels
      
def hz2semitone(hz):
    return 69 + 12 * np.log2(hz/440.)

def hz2mel(hz):
    """Convert a value in Hertz to Mels
    :param hz: a value in Hz. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Mels. If an array was passed in, an identical sized array is returned.
    """
    return 2595 * np.log10(1+hz/700.)

def mel2hz(mel):
    """Convert a value in Mels to Hertz
    :param mel: a value in Mels. This can also be a numpy array, conversion proceeds element-wise.
    :returns: a value in Hertz. If an array was passed in, an identical sized array is returned.
    """
    return 700*(10**(mel/2595.0)-1)

def get_filterbanks(nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
    """Compute a Mel-filterbank. The filters are stored in the rows, the columns correspond
    to fft bins. The filters are returned as an array of size nfilt * (nfft/2 + 1)
    :param nfilt: the number of filters in the filterbank, default 20.
    :param nfft: the FFT size. Default is 512.
    :param samplerate: the samplerate of the signal we are working with. Affects mel spacing.
    :param lowfreq: lowest band edge of mel filters, default 0 Hz
    :param highfreq: highest band edge of mel filters, default samplerate/2
    :returns: A numpy array of size nfilt * (nfft/2 + 1) containing filterbank. Each row holds 1 filter.
    """
    highfreq= highfreq or samplerate/2
    assert highfreq <= samplerate/2, "highfreq is greater than samplerate/2"

    # compute points evenly spaced in mels
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilt+2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = np.floor((nfft+1)*mel2hz(melpoints)/samplerate)

    fbank = np.zeros([nfilt,nfft//2+1])
    for j in range(0,nfilt):
        for i in range(int(bin[j]), int(bin[j+1])):
            fbank[j,i] = (i - bin[j]) / (bin[j+1]-bin[j])
        for i in range(int(bin[j+1]), int(bin[j+2])):
            fbank[j,i] = (bin[j+2]-i) / (bin[j+2]-bin[j+1])
    return fbank

def getTestFeat(infile, mode):
    frames = Signal(infile, mode=3)
    features = frames.pickFeatures(pick=(1,1,1,1,1))
    
    return features

if __name__ == '__main__':
  
    feat = Signal('03a01Wa.wav')
    print(feat.formant())


