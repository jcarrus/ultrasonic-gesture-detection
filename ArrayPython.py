import serial
from scipy.signal import butter, lfilter
import random
import time
import numpy as np
import matplotlib.pyplot as plt

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def generate_signal(length = 8192, pad = 200, fs = 130000):
    random_signal = np.array(np.random.normal(0, 1, length - (2 * pad)))
    filtered_signal = butter_bandpass_filter(random_signal, 2000.0, 40000.0, fs)
    signal  = filtered_signal - np.mean(filtered_signal)
    signal[signal <= 0] = -1
    signal[signal > 0] = 1
    padding = np.array([0 for x in range(pad)])
    signal = np.concatenate((padding, signal, padding))
    signal_binary = np.array(signal > 0, dtype=int)
    return (signal, signal_binary)

def send_signal(ser, signal):
    print('Signal: ', signal)
    print('Signal Length: ', len(signal))
    signal_length = len(signal).to_bytes(2, byteorder='big')
    print('Sending as signal_length: ', signal_length)
    ser.write(signal_length)
    ser.write(np.packbits(signal).tobytes())
    ser.flush()

def read_teensy(ser):
    num_bytes = 2
    num_samplers = 2
    num_samples = int.from_bytes(ser.read(2), byteorder='big')
    print('Took ', num_samples, ' samples')

    sample_period = int.from_bytes(ser.read(4), byteorder='big')/1000000
    sample_freq = num_samples / sample_period
    print('Sampling Freq: ', sample_freq)

    raw_output = ser.read(num_samples * num_bytes * num_samplers)
    output = [(raw_output[x] << 8) + raw_output[x+1] for x in range(0, len(raw_output), 2)]
    out1 = np.array(output[:len(output)//2])
    out1 = np.subtract(out1, np.mean(out1))
    out2 = np.array(output[len(output)//2:])
    out2 = np.subtract(out2, np.mean(out2))
    return (sample_freq, out1, out2)
    
def discover_system(i, o):
    sys = np.fft.ifft(np.fft.rfft(o) / np.fft.rfft(i))
    return sys

def classify_system(s1, s2):
    pass

if __name__ == "__main__":
    with serial.Serial('/dev/ttyACM0', 19200) as ser:

        # Clear any preexisting serial messages
        print(ser.inWaiting(), ' bytes in waiting')
        while ser.inWaiting() > 0:
            ser.read(1)

        # Generate a new signal and send it to the teensy
        sig, sig_binary = generate_signal(length=8192, pad=600)
        send_signal(ser, sig_binary)

        # Read the output from the teensy
        fs, out1, out2 = read_teensy(ser)
        
        # Discover the systems
        sys1 = discover_system(sig, out1)
        sys2 = discover_system(sig, out2)

        # Actually try to classify the systems
        classify_system(sys1, sys2)

        # Plot things for mic 1
        fig = plt.figure(1)
        ax = plt.subplot(321)
        ax.set_title('Input Signal')
        plt.plot(np.array(range(len(sig))) / fs, sig)
        plt.xlabel('Time (s)')
        plt.ylabel('Signal')
        ax = plt.subplot(322)
        ax.set_title('Input Signal FFT')
        plt.plot(np.fft.rfftfreq(len(sig), d=1/fs), np.fft.rfft(sig))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        #plt.xlim([0, 750])
        ax = plt.subplot(323)
        ax.set_title('Mic Raw Signal')
        plt.plot(np.array(range(len(out1))) / fs, (out1/(2**12) * 3.3))
        plt.xlabel('Time (s)')
        plt.ylabel('Voltage')
        ax = plt.subplot(324)
        ax.set_title('Mic Raw Signal FFT')
        plt.plot(np.fft.rfftfreq(len(out1), d=1/fs), np.fft.rfft(out1/(2**12) * 3.3))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Power')
        # plt.xlim([30000, 60000])
        ax = plt.subplot(325)
        ax.set_title('System')
        plt.plot(np.array(range(len(sys1))) / fs, sys1)
        plt.xlabel('Lag (s)')
        plt.ylabel('Response')
        ax = plt.subplot(326)
        # plt.xlim([30000, 60000])
        ax.set_title('System FFT')
        plt.plot(np.fft.rfftfreq(len(sys1), d=1/fs), np.fft.rfft(sys1))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Gain')
        plt.tight_layout()

        plt.figure(2)
        plt.plot((np.array(range(len(sig) * 2 - 1))-(len(sig))) / fs, np.correlate(sig, out1, 'full'))
        plt.xlabel('Lag (s)')
        plt.ylabel('Power')
        plt.title('Input-Output Cross Correlation')
        plt.tight_layout()
        
        plt.show()
        
