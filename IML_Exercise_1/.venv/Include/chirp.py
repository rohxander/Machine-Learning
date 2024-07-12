
import numpy as np

def createChirpSignal(samplingrate: int, duration: int, freqfrom: int, freqto: int, linear: bool):
    sample_array = np.linspace(0, duration, duration * samplingrate)
    if(linear == True):
        sample_array = np.sin(2*np.pi*(freqfrom + ((freqto - freqfrom)*0.5*sample_array/duration))*sample_array)
        return sample_array
    else:

        x = (freqfrom*duration)/(np.log(freqto/freqfrom))
        sample_array = np.sin(2*np.pi*x*(((freqto/freqfrom)**(sample_array/duration))-1))
        return sample_array
