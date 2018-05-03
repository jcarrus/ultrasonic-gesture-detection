import serial
from scipy.signal import butter, lfilter
import random
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
import json
import glob
import os

# A tool to help print things
def p(debug, object):
    if debug == 'true':
        print(object)

# Process the command line arguments and return an args dictionary
def process_args():
    args = {'length': 8192,
            'padding': 200,
            'signal_input': '',
            'signal_type': '',
            'serial_port': '',
            'use_datafile': '',
            'save_output': '',
            'debug': ''}
    for arg in sys.argv:
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
        if arg[:6] == 'debug=':
            args['debug'] = arg[6:]
            print('Detected argument debug: ', args['debug'])
    return args

# Ensure that the directory exists
def ensure_dir(file_path):
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

# Lists the serial ports
def list_serial_ports():
    if sys.platform.startswith('win'):
        ports = ['COM%s' % (i + 1) for i in range(256)]
    elif sys.platform.startswith('linux') or sys.platform.startswith('cygwin'):
        # this excludes your current terminal "/dev/tty"
        ports = glob.glob('/dev/tty[A-Za-z]*')
    elif sys.platform.startswith('darwin'):
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
            print(sys.exc_info()[0])
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

# Generate a binary stochastic signal of the length `length`
def generate_stochastic_signal(length, padding):
    random_signal = np.array(np.random.normal(0, 1, length - (2 * padding)))
    filtered_signal = butter_bandpass_filter(random_signal, 2000.0, 40000.0, 130000)
    signal  = filtered_signal - np.mean(filtered_signal)
    signal[signal <= 0] = -1
    signal[signal > 0] = 1
    padding = np.array([0 for x in range(padding)])
    signal = np.concatenate((padding, signal, padding))
    signal_binary = np.array(signal > 0, dtype=int)
    return (signal, signal_binary)

# Generate a square signal of the length `length` padded with `padding` zeros
def generate_square_signal(length, padding):
    signal  = np.resize([1,1,1,1, -1,-1,-1,-1], length - (2 * padding))
    padding = np.resize([0], padding)
    signal = np.concatenate((padding, signal, padding))
    signal_binary = np.array(signal > 0, dtype=int)
    return (signal, signal_binary)

# Generate a single square signal of the length `length` padded with `padding` zeros
def generate_single_square_signal(length, padding):
    signal  = np.resize([1], length - (2 * padding))
    padding = np.resize([0], padding)
    signal = np.concatenate((padding, signal, padding))
    signal_binary = np.array(signal > 0, dtype=int)
    return (signal, signal_binary)

# Send the `signal` over the serial `ser`
def send_signal(ser, signal):
    print('Signal: ', signal)
    print('Signal Length: ', len(signal))
    signal_length = len(signal).to_bytes(2, byteorder='big')
    print('Sending as signal_length: ', signal_length)
    ser.write(signal_length)
    ser.write(np.packbits(signal).tobytes())
    ser.flush()

# Read in a teensy datastring from serial `ser`
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

# Discover the system from an input `i` and output `o`
def discover_system(i, o):
    padding = np.resize([0], len(i))
    i_padded = np.concatenate((padding, i, padding))
    o_padded = np.concatenate((padding, o, padding))
    ffts = np.divide(np.fft.fft(o_padded), np.fft.fft(i_padded))
    sys = np.real(np.fft.ifft(ffts[:ffts.size]))
    return sys

# Classify the identified systems as a gesture state
def classify_system(s1, s2):
    pass

## The main function
if __name__ == "__main__":

    # Process the command line arguments
    args = process_args()
    debug = args['debug']
    p(debug, args)
    
    #####################
    # Generate a Signal #
    #####################
    # If we pass in a signal file...
    if not args['signal_input'] == '':
        # Open that signal file...
        filename = './data/%s.datafile' % args['signal_input']
        with open(filename) as f:
            # And load that to memory...
            data = json.load(f)
            sig = data['sig']
            sig_binary = data['sig_binary']
    # Otherwise...
    elif args['signal_type'] == 'square':
        # Generate a square signal
        sig, sig_binary = generate_square_signal(args['length'], args['padding'])
    elif args['signal_type'] == 'single_square':
        # Generate a square signal
        sig, sig_binary = generate_single_square_signal(args['length'], args['padding'])
    elif args['signal_type'] == 'stochastic':
        # Generate a new binary stochastic signal
        sig, sig_binary = generate_stochastic_signal(args['length'], args['padding'])
    else:
        print('Signal type not specified, defaulting to stochastic')
        sig, sig_binary = generate_stochastic_signal(args['length'], args['padding'])
    
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
            send_signal(ser, sig_binary)
            # Read the output from the teensy...
            fs, out1, out2 = read_teensy(ser)
    # Otherwise, read the datafile...
    else:
        # Open the data file...
        filename = './data/%s/%s.datafile' % (args['use_datafile'], args['use_datafile'])
        with open(filename, 'r') as f:
            # And load it to memory...
            data = json.load(f)
            fs = data['fs']
            out1 = np.array(data['out1'])
            out2 = np.array(data['out2'])

    #################
    # Save the Data #
    #################
    if not args['save_output'] == '':
        filename = './data/%s/%s.datafile' % (args['save_output'], args['save_output'])
        ensure_dir(filename)
        with open(filename, 'w') as f:
            json.dump({'sig': sig.tolist(),
                       'sig_binary': sig_binary.tolist(),
                       'fs': fs,
                       'out1': out1.tolist(),
                       'out2': out2.tolist()}, f)
            print('Saved data to: ', filename)

    ########################
    # Discover the systems #
    ########################
    sys1 = discover_system(sig, out1)
    sys2 = discover_system(sig, out2)

    ####################################
    # Classify the system as a gesture #
    ####################################
    classify_system(sys1, sys2)

    ###############
    # Plot things #
    ###############
    # figure 1 shows how to discover the system for mic 1
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
    ax = plt.subplot(325)
    ax.set_title('System')
    plt.plot(np.array(range(len(sys1))) / fs, sys1)
    plt.xlabel('Lag (s)')
    plt.ylabel('Response')
    ax = plt.subplot(326)
    ax.set_title('System FFT')
    plt.plot(np.fft.rfftfreq(len(sys1), d=1/fs), np.fft.rfft(sys1))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Gain')
    plt.tight_layout()
    if not args['save_output'] == '':
        plt.savefig('./data/%s/%s - 1.png' % (args['save_output'], args['save_output']), dpi=600, format='png')
    
    # Figure 2 is an input/output correlation plot
    plt.figure(2)
    plt.plot((np.array(range(len(sig) * 2 - 1))-(len(sig))) / fs, np.correlate(out1, sig, 'full'))
    plt.xlabel('Lag (s)')
    plt.ylabel('Power')
    plt.title('Input-Output Cross Correlation')
    plt.tight_layout()
    if not args['save_output'] == '':
        plt.savefig('./data/%s/%s - 2.png' % (args['save_output'], args['save_output']), dpi=600, format='png')

    # Figure 3 is an input/output correlation plot
    plt.figure(3)
    plt.plot((np.array(range(len(sig) * 2 - 1))-(len(sig))) / fs, np.correlate(out1, np.concatenate((np.diff(sig), np.array([0]))), 'full'))
    plt.xlabel('Lag (s)')
    plt.ylabel('Power')
    plt.title('Input-Output Cross Correlation')
    plt.tight_layout()
    if not args['save_output'] == '':
        plt.savefig('./data/%s/%s - 3.png' % (args['save_output'], args['save_output']), dpi=600, format='png')
    plt.xlim(0, 0.004)
    if not args['save_output'] == '':
        plt.savefig('./data/%s/%s - 4.png' % (args['save_output'], args['save_output']), dpi=600, format='png')

    # Show all of our plots
    plt.show()
    
