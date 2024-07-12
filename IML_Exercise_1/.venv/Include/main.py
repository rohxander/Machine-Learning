from chirp import createChirpSignal
from decomposition import createTriangleSignal, createSquareSignal, createSawtoothSignal
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import chirp

samplers = np.linspace(0, 1, 1*200)
chirp_array = createChirpSignal( 200 , 1 , 1 , 10 , True)

samples = 200
frequency = 2
kMax = 10000
amplitude = 1
triangle_signal = createTriangleSignal(samples, frequency, kMax)
square_signal = createSquareSignal(samples, frequency, kMax)
sawtooth_signal = createSawtoothSignal(samples, frequency, kMax, amplitude)

t = np.linspace(0, 1, samples)
# plt.figure()
# plt.plot(samplers, chirp_array)
# plt.title('Linear Chirp Signal')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.show()

plt.figure(figsize=(12, 6))

plt.subplot(4, 1, 1)
plt.plot(t,triangle_signal)
plt.title('Triangle Signal')
plt.xlabel('Samples')
plt.ylabel('Amplitude')

plt.subplot(4, 1, 2)
plt.plot(t,square_signal)
plt.title('Square Signal')
plt.xlabel('Samples')
plt.ylabel('Amplitude')

plt.subplot(4, 1, 3)
plt.plot(t,sawtooth_signal)
plt.title('Sawtooth Signal')
plt.xlabel('Samples')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()
# TODO: Test the functions imported in lines 1 and 2 of this file.
