from __future__ import print_function
from __future__ import division
import math
import wave
import struct
import contextlib
import webrtcvad
import numpy as np
import matplotlib.pyplot as plt

 
def read_data(path):
    """Reads a .wav file.
    Takes the path, and returns (signal, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000, 48000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate

"""
signal input: PCM wav format
for voice activity detection
"""

class Frame(object):
    """Represents a "frame" of audio data."""
    def __init__(self, bytes):
        self.bytes = bytes
        

def frame_generator(frame_duration_ms,audio,sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n])
        offset += n

 
def getVoicedFrames(audio,mode,win_ms,sample_rate):
    vad = webrtcvad.Vad(mode)
    frames = frame_generator(win_ms, audio, sample_rate)
    voiced_frames = []
    for frame in frames:
        if vad.is_speech(frame.bytes,sample_rate) == True:
            voiced_frames.append(frame)
    yield b''.join([f.bytes for f in voiced_frames])

  
def getVoicedSignal(audio,sample_rate,win_ms=30,mode=2):
    voiced_frames = getVoicedFrames(audio,mode,win_ms,sample_rate)
    for frame in voiced_frames:
        temp = struct.unpack("%ih" % (len(frame) // 2), frame)
        voiced_signal = [float(val) for val in temp]
   
    voiced_signal = np.asarray(voiced_signal)
    assert np.max(voiced_signal) != 0
    voiced_signal = np.double(voiced_signal)
    voiced_signal = voiced_signal / 2.0 ** 15
    DC = np.mean(voiced_signal)
    MAX = (np.abs(voiced_signal)).max()
    voiced_signal = (voiced_signal - DC) / (MAX + 1e-10)

    return voiced_signal

####################################################################

"""
signal format:float

"""

def pre_emphasis(signal,coeff=0.97):
    return np.append(signal[0],signal[1:] - coeff*signal[:-1])


def framing(signal,sample_rate,win_ms=0.025,overlap=0.015,preemp=0.97):
    emphasized_signal = pre_emphasis(signal, preemp)
    frame_length, frame_step = sample_rate * win_ms, sample_rate * (win_ms - overlap)
    signal_length = len(signal)
    frame_length = int(round(frame_length))
    frame_step = int(round(frame_step))
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)) ## at least one frame
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized_signal, z)

    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]

    frames *= np.hamming(frame_length) #use hamming function
    return frames

"""
from python_speech_feature
"""
def magspec(frames,NFFT):
    """Compute the magnitude spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).
    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the magnitude spectrum of the corresponding frame.
    """
    if np.shape(frames)[1] > NFFT:
        raise Exception('frame length (%d) is greater than FFT size (%d), frame will be truncated. Increase NFFT to avoid.', np.shape(frames)[1], NFFT)
    complex_spec = np.fft.rfft(frames, NFFT)
    return np.absolute(complex_spec)


def powspec(frames,NFFT):
    """Compute the power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).
    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the power spectrum of the corresponding frame.
    """
    return 1.0 / NFFT * np.square(magspec(frames, NFFT))


def logpowspec(frames,NFFT,norm=1):
    """Compute the log power spectrum of each frame in frames. If frames is an NxD matrix, output will be Nx(NFFT/2+1).
    :param frames: the array of frames. Each row is a frame.
    :param NFFT: the FFT length to use. If NFFT > frame_len, the frames are zero-padded.
    :param norm: If norm=1, the log power spectrum is normalised so that the max value (across all frames) is 0.
    :returns: If frames is an NxD matrix, output will be Nx(NFFT/2+1). Each row will be the log power spectrum of the corresponding frame.
    """
    ps = powspec(frames, NFFT);
    ps[ps <= 1e-30] = 1e-30
    lps = 10 * np.log10(ps)
    if norm:
        return lps - np.max(lps)
    else:
        return lps


if __name__ == '__main__':
    path = 'chunk-01.wav'
    audio, sample_rate = read_data(path)
    voiced_signal = getVoicedSignal(audio, sample_rate, 30, 2)
    #voiced_signal = voiced_signal[0:int(sample_rate*3.5)]
    frames = framing(voiced_signal, sample_rate)
    plt.plot(voiced_signal, c = 'b')
    #plt.plot(pre_e,c = 'r')
    plt.show()
