import numpy as np

def createTriangleSignal(samples: int, frequency: int, k_max: int):
    t = np.linspace(0, 1, samples)
    signal = np.zeros(samples)
    for k in range(1, k_max+1):
        bn = 8 * ((-1) ** (k + 1)) / ((2 * k - 1) ** 2 * np.pi ** 2)
        value = bn * np.sin(2 * np.pi * (2 * k - 1) * frequency * t)
        signal += value
    return signal

def createSquareSignal(samples: int, frequency: int, k_max: int):
    t = np.linspace(0, 1, samples)
    signal = np.zeros(samples)
    for k in range(1, k_max + 1):
        bn = 4 / np.pi / (2 * k - 1)
        value =  bn * np.sin(2 * np.pi * (2 * k - 1) * frequency * t)
        signal += value
    return signal


def createSawtoothSignal(samples: int, frequency: int, k_max: int, amplitude: int):
    t = np.linspace(0, 1, samples)
    signal = np.zeros(samples)
    a0 = amplitude / 2
    for k in range(1, k_max):
        bn = ((-1)*(amplitude / (k*np.pi)))
        value = (bn * (np.sin(2 * np.pi * k * frequency * t)))
        signal += value
    signal+=a0
    return signal

