import serial
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import sys
import json
import glob
import os
from scipy.signal import butter, lfilter, hilbert, convolve, deconvolve, correlate
from scipy.linalg import toeplitz, solve_toeplitz

# A tool to help print things
def p(debug, object):
    if debug == True:
        print(object)

help_string = """
'ello dere m'ate. Hehz ya optshions fa diz'ere funkshun.

usage:

$ python main.py [help] [option1=option_text option2=option_text ...]

options:

help:                 print this useful help statement
realtime:             do things continuously
signal_input=dataset: use the signal previously generated and stored in a dataset
signal_type=type:     generate a signal of a particular type [simple_stochasic]
serial_port=path:     use the serial port specified in path
use_datafile=dataset: use the input and mic output data from a particular dataset
save_output=dataset:  save the output of the script to a dataset with name dataset
use_toeplitz:         use the toeplitz form of deconvolution
filter_mic:           run the mic output through a bandpass filter to remove some noise
debug:                print lots of things to help debug this script
"""

# Process the command line arguments and return an args dictionary
def process_args():
    args = {'length': 32768,
            'padding': 200,
            'signal_input': '',
            'signal_type': '',
            'serial_port': '',
            'use_datafile': '',
            'save_output': '',
            'num_repeats': 1,
            'build_signals': 0,
            'use_toeplitz': False,
            'random_signals': False,
            'convolve_input': False,
            'filter_mic': False,
            'realtime': False,
            'debug': False}
    for arg in sys.argv:
        if arg == 'help':
            print(help_string)
            exit()
        if arg[:7] == 'length=':
            args['length'] = int(arg[7:])
            print('Detected argument length: ', args['length'])
        if arg[:8] == 'padding=':
            args['padding'] = int(arg[8:])
            print('Detected argument padding: ', args['padding'])
        if arg[:13] == 'signal_input=':
            args['signal_input'] = arg[13:]
            print('Detected argument signal_input: ', args['signal_input'])
        if arg[:12] == 'signal_type=':
            args['signal_type'] = arg[12:]
            print('Detected argument signal_type: ', args['signal_type'])
        if arg[:12] == 'serial_port=':
            args['serial_port'] = arg[12:]
            print('Detected argument serial_port: ', args['serial_port'])
        if arg[:13] == 'use_datafile=':
            args['use_datafile'] = arg[13:]
            print('Detected argument use_datafile: ',args['use_datafile'])
        if arg[:12] == 'save_output=':
            args['save_output'] = arg[12:]
            print('Detected argument save_output: ', args['save_output'])
        if arg[:12] == 'num_repeats=':
            args['num_repeats'] = int(arg[12:])
            print('Detected argument num_repeats: ', args['num_repeats'])
        if arg[:14] == 'build_signals=':
            args['build_signals'] = int(arg[14:])
            print('Detected argument build_signals: ', args['build_signals'])
        if arg == 'random_signals':
            args['random_signals'] = True
            print('Detected argument random_signals')
        if arg == 'use_toeplitz':
            args['use_toeplitz'] = True
            print('Detected argument use_toeplitz')
        if arg == 'filter_mic':
            args['filter_mic'] = True
            print('Detected argument filter_mic')
        if arg == 'convolve_input':
            args['convolve_input'] = True
            print('Detected argument convolve_input')
        if arg == 'realtime':
            args['realtime'] = True
            print('Detected argument realtime')
        if arg[:6] == 'debug':
            args['debug'] = True
            print('Detected argument debug: ', args['debug'])
    return args

# Ensure that the directory exists
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

# Save data to file
def save_data(filename, data):
    ensure_dir(filename)
    with open(filename, 'w') as f:
        json.dump(data, f)
        print('Saved data to: ', filename)

# Lists the serial ports
def list_serial_ports():
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
        # TODO Figure out why this doesn't work for Julia
        ports = glob.glob('/dev/tty.*')
    else:
        raise EnvironmentError('Unsupported platform')

    result = []
    for port in ports:
        try:
            s = serial.Serial(port)
            s.close()
            result.append(port)
        except (OSError, serial.SerialException):
            pass
    return result

# A helper function for the filter below
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

# Filter data with a bandpass filter
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_lowpass(cutoff, fs, order=5):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return b, a

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Generate a binary stochastic signal of the length `length`
def generate_stochastic_signal(length):
    signal = np.resize([1, 0], length)
    for i in range(1, length):
        if random.random() > 0.995:
            signal[i] = -1 * signal[i-1]
        else:
            signal[i] = signal[i-1]
    return signal.tolist()

# Generate a single square signal of the length `length` padded with `padding` zeros
def generate_single_square_signal(length, padding):
    signal  = np.resize([1], length - (2 * padding))
    padding = np.resize([0], padding)
    signal = np.concatenate((padding, signal, padding))
    return signal

# Send the `signal` over the serial `ser`
def send_signal(ser, signal):
    # Scale the signal for the teensy
    s = np.array(signal) - np.mean(signal)
    s *= (255.0 / 2) / abs(s).max()
    s += (255.0 / 2)
    s = np.clip(s, 0, 1)
    ser.write(bytes([int(x) for x in s]))
    ser.flush()

# Read in a teensy datastring from serial `ser`
def read_teensy(ser, length):
    num_bytes = 2
    num_samplers = 2
    raw_output = ser.read(length * num_bytes * num_samplers)
    output = [(raw_output[x] << 8) + raw_output[x+1] for x in range(0, len(raw_output), 2)]
    times = np.array(output[:len(output)//2])
    out1 = np.array(output[len(output)//2:])
    out1 = out1 - np.mean(out1)
    out2 = np.copy(out1)
    return (times.tolist(), out1.tolist(), out2.tolist())

# Test for audio amplifier failure
def verify_systems(systems):
    powers = np.max(np.abs(systems), axis=1)
    print(powers.tolist())
    print('Valid systems are %s percent' % (np.sum(powers > 10) * 100 / len(powers)))
    print(powers.tolist())

# Discover the system from an input `i` and output `o` using a traditional FFT method
def discover_system(i, o):
    padding = np.resize([0], len(i))
    i_padded = np.concatenate((i, padding))
    o_padded = np.concatenate((o, padding))
    i_fft = np.fft.fft(i_padded)
    np.place(i_fft, i_fft == 0, [.01])
    o_fft = np.fft.fft(o_padded)
    ffts = np.divide(o_fft, i_fft)
    sys = np.fft.ifft(ffts)
    return sys

# Discover the system from an input `i` and output `o` using a toeplitz form
def discover_system_toeplitz(i, o, fs):
    mean_i = np.mean(i)
    mean_o = np.mean(o)
    num_samples = len(i)
    max_lag = num_samples // 10
    Cxx = np.resize([0], max_lag)
    Cxy = np.resize([0], max_lag)
    for j in range(max_lag):
        Cxx[j] = np.sum([(i[k-j] - mean_i) * (i[k] - mean_i) for k in range(j, num_samples)])
        Cxy[j] = np.sum([(i[k-j] - mean_i) * (o[k] - mean_o) for k in range(j, num_samples)])

    col = np.resize([0], len(Cxx))
    col[0] = Cxx[0]
    
    sys = fs * solve_toeplitz((col, Cxx), Cxy) / num_samples
    sys = moving_abs_average(sys, 0)
    return sys


# A model to fit our system
def model(x, a, b, c, d, e, f):
    return np.multiply(a*(1-np.exp(-(x-f)/b))*(np.exp(-(x-f)/c))*np.sin(2*np.pi*((x-f)-e)/d), [bool(i > f) for i in range(len(x))])

# fit the system to our model
def fit_system(sys):
    pass

# Classify the identified systems as a gesture state
def classify_system(s1):
    pass

# Update the plots for the realtime viewer.
def update_plot(times, sig, out1, sys1, out2, sys2, fig, l0, l1, l2, l3, l4, l5):

    fs = 1000000 / (np.mean(np.diff(times)))
    # figure 1 shows how to discover the system for mic 1
    l0.set_xdata(times/1000000)
    l0.set_ydata(sig)
    l1.set_xdata(np.fft.rfftfreq(len(sig), d=1/fs))
    l1.set_ydata(np.fft.rfft(sig))
    l2.set_xdata(times / 1000000)
    l2.set_ydata((out1/(2**12) * 3.3))
    l3.set_xdata(np.fft.rfftfreq(len(out1), d=1/fs))
    l3.set_ydata(np.fft.rfft(out1/(2**12) * 3.3))
    l4.set_xdata(np.array(range(len(sys1))) / fs)
    l4.set_ydata(sys1)
    l5.set_xdata(np.fft.rfftfreq(len(sys1), d=1/fs))
    l5.set_ydata(abs(np.fft.rfft(sys1)))
    fig.canvas.draw()
    
# Make plots for visualizing data
def plot(times, sig, out1, sys1, out2, sys2):
    fs = 1000000 / (np.mean(np.diff(times)))
    # figure 1 shows how to discover the system for mic 1
    fig = plt.figure(1, figsize=(8,8))
    ax = plt.subplot(321)
    ax.set_title('Input Signal')
    l0, = plt.plot(times/1000000, sig)
    #plt.xlim(0, .005)
    plt.xlabel('Time (s)')
    plt.ylabel('Signal')
    ax = plt.subplot(322)
    ax.set_title('Input Signal FFT')
    l1, = plt.plot(np.fft.rfftfreq(len(sig), d=1/fs), np.fft.rfft(sig))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    ax = plt.subplot(323)
    ax.set_title('Mic Raw Signal')
    l2, = plt.plot(times / 1000000, (out1/(2**12) * 3.3), 'k.')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage')
    #plt.xlim(0, .005)
    plt.ylim(-2, 2)
    ax = plt.subplot(324)
    ax.set_title('Mic Raw Signal FFT')
    l3, = plt.plot(np.fft.rfftfreq(len(out1), d=1/fs), np.fft.rfft(out1/(2**12) * 3.3))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    ax = plt.subplot(325)
    ax.set_title('System')
    l4, = plt.plot(np.array(range(len(sys1))) / fs, sys1)
    plt.xlabel('Lag (s)')
    plt.ylabel('Response')
    plt.xlim(0, .005)
    #plt.ylim(-3, 3)
    ax = plt.subplot(326)
    ax.set_title('System FFT')
    l5, = plt.loglog(np.fft.rfftfreq(len(sys1), d=1/fs), abs(np.fft.rfft(sys1)))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    #plt.xlim(500, 130000)
    #plt.ylim(10, 100000)
    plt.tight_layout()
    fig.canvas.draw()
    if not args['save_output'] == '':
        plt.savefig('./data/%s/%s - 1.png' % (args['save_output'], args['save_output']), dpi=600, format='png')
    
    # # Figure 2 is an input/output correlation plot
    # plt.figure(2)
    # plt.plot((np.array(range(len(sig) * 2 - 1))-(len(sig))) / fs, np.correlate(out1, sig, 'full'))
    # plt.xlabel('Lag (s)')
    # plt.ylabel('Power')
    # plt.title('Input-Output Cross Correlation')
    # plt.tight_layout()
    # if not args['save_output'] == '':
    #     plt.savefig('./data/%s/%s - 2.png' % (args['save_output'], args['save_output']), dpi=600, format='png')

    # # Figure 3 is an input/output correlation plot
    # plt.figure(3)
    # plt.plot((np.array(range(len(sig) * 2 - 1))-(len(sig))) / fs, np.correlate(out1, np.concatenate((np.diff(sig), np.array([0]))), 'full'))
    # plt.xlabel('Lag (s)')
    # plt.ylabel('Power')
    # plt.title('Input-Output Cross Correlation')
    # plt.tight_layout()
    # plt.xlim(0, 0.004)
    # if not args['save_output'] == '':
    #     plt.savefig('./data/%s/%s - 4.png' % (args['save_output'], args['save_output']), dpi=600, format='png')

    return fig, (l0, l1, l2, l3, l4, l5)

def run_system(args):
    debug = args['debug']
    num_repeats = args['num_repeats']
    p(debug, args)
    
    #####################
    # Generate a Signal #
    #####################
    # If we don't pass in a datafile...
    if args['use_datafile'] == '':
        # If we pass in a signal file...
        if not args['signal_input'] == '':
            # Open that signal file...
            filename = './data/%s/%s.datafile' % (args['signal_input'], args['signal_input'])
            with open(filename) as f:
                # And load that to memory...
                data = json.load(f)
                sig = data['sig']
                if args['random_signals']:
                    np.random.shuffle(sig)
                if len(sig) < args['num_repeats'] and not args['random_signals']:
                    print('Not enough signals for the number of repeats')
                    print('Will wrap signal')
                sig = np.resize(sig, (args['num_repeats'], len(sig[0])))
        # Otherwise...
        else:
            sig = []
            if args['signal_type'] == 'stochastic':
                # Generate a new binary stochastic signal
                for i in range(num_repeats):
                    sig.append(generate_stochastic_signal(args['length']))
            else:
                print('Signal type not specified, defaulting to stochastic')
                for i in range(num_repeats):
                    sig.append(generate_stochastic_signal(args['length']))

    #####################
    # Get a data stream #
    #####################
    # If we did not pass in data file, poll the teensy for new data
    if args['use_datafile'] == '':
        # If we didn't specify a port, then try to find a port
        if args['serial_port'] == '':
            ports = list_serial_ports()
            if len(ports) == 0:
                raise RuntimeError('No Serial ports found')
            else:
                args['serial_port'] = ports[0]
        # Open the serial port...
        with serial.Serial(args['serial_port'], 19200) as ser:
            # Clear any preexisting serial messages...
            p(debug, '%s bytes in waiting' % ser.inWaiting() )
            while ser.inWaiting() > 0:
                ser.read(1)
            # Send the signal...
            times = []
            out1 = []
            out2 = []
            for i in range(num_repeats):
                send_signal(ser, sig[i])
                # Read the output from the teensy...
                t, o1, o2 = read_teensy(ser, len(sig[i]))
                times.append(t)
                out1.append(o1)
                out2.append(o2)
    # Otherwise, read the datafile...
    else:
        # Open the data file...
        filename = './data/%s/%s.datafile' % (args['use_datafile'], args['use_datafile'])
        with open(filename, 'r') as f:
            # And load it to memory...
            data  = json.load(f)
            times = data['times']
            out1  = data['out1']
            out2  = data['out2']
            sig   = data['sig']


    ########################
    # Discover the systems #
    ########################

    sys1 = []
    sys2 = []
    
    for i in range(num_repeats):
        fs = 1000000 / (np.mean(np.diff(times[i])))
        s = sig[i]
        o1 = out1[i]
        o2 = out2[i]
        
        print('Iteration: ', i, ' with sampling frequency of: ', fs)

        if args['filter_mic'] == True:
            print('WARNING: filtering the mic is not recommended')
            o1 = butter_bandpass_filter(o1, 30000, 50000, fs)
            o2 = butter_bandpass_filter(o2, 30000, 50000, fs)
        
        if args['convolve_input']:
            sys = json.load(open('./data/transducer_id/system.sys', 'r'))
            s = convolve(s, sys, mode='full')[:len(s)]
            # # Test for non-linearity
            # plt.figure(3)
            # plt.plot(s, o1, '.')
            # plt.xlabel('Predicted System Output')
            # plt.ylabel('Actual System Output')
            # plt.title('Non-linearity Detection')
            # plt.savefig('./data/transducer_id/non-linearity_detection.png', dpi=600, format='png')
            # plt.show()

            y_bar = np.mean(o1)
            SStot = np.sum((o1 - y_bar)**2)
            SSres = np.sum((o1 - s)**2)
            r_squared = 1 - (SSres / SStot)
            print('Variance accounted for: ', r_squared)
            
        if args['use_toeplitz'] == True:
            sys1.append(discover_system_toeplitz(s, o1, fs))
        else:
            sys1.append(discover_system(s, o1))
            p(debug, sys1)

    times = np.array(times[0])
    sig = np.array(s)
    out1 = np.array(o1)
    out2 = np.array(o2)
    sys1 = np.real(np.mean(sys1, axis=0))
    sys2 = np.copy(sys1)

    plt.figure(4)
    h = json.load(open('./data/transducer_id/system.sys', 'r'))
    d = correlate(h, sys1)
    print(d.tolist())
    plt.plot(d)
    plt.show()
    
    #################
    # Save the Data #
    #################
    if not args['save_output'] == '' and args['build_signals'] == 0:
        filename = './data/%s/%s.datafile' % (args['save_output'], args['save_output'])
        save_data(filename, {'sig': [sig.tolist()],
                             'times': [times.tolist()],
                             'out1': [out1.tolist()],
                             'out2': [out2.tolist()],
                             'sys1': sys1.tolist(),
                             'sys2': sys2.tolist()})
        
    ####################################
    # Classify the system as a gesture #
    ####################################
    classify_system(sys1)

    return (times, sig, out1, sys1, out2, sys2)

## The main function
if __name__ == "__main__":

    # Process the command line arguments
    args = process_args()

    if not args['build_signals'] == 0:
        if args['save_output'] == '':
            sys.exit('No output path specified. Use the save_output flag to specify')
        filename = './data/%s/%s.datafile' % (args['save_output'], args['save_output'])
        good_signals = []
        while len(good_signals) < args['build_signals']:
            times, sig, out1, sys1, out2, sys2 = run_system(args)
            if np.max(np.abs(sys1)) > 10:
                good_signals.append(sig)
                print('Found good signal')
        save_data(filename, {'sig': [sample.tolist() for sample in good_signals]})
        print('Signals complete')
    
    elif args['realtime']:
        fig, l = plot(*run_system(args))
        print(l)
        plt.show(block=False)
        while(1):
            update_plot(*run_system(args), fig, *l)
    else:
        plot(*run_system(args))
        plt.show()
