import numpy as np


def sinusoidal_signal(frequency, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sin(2 * np.pi * frequency * t)

def square_wave_signal(frequency, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return np.sign(np.sin(2 * np.pi * frequency * t))

def sawtooth_wave_signal(frequency, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return 2 * (t * frequency - np.floor(t * frequency + 0.5))

def triangle_wave_signal(frequency, duration, sample_rate):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    return 2 * np.abs(2 * (t * frequency - np.floor(t * frequency + 0.5))) - 1


if __name__ == "__main__":
    from matplotlib import pyplot as plt
    frequency = 5  # Hz
    duration = 1   # seconds
    sample_rate = 100  # samples per second

    sinusoidal = sinusoidal_signal(frequency, duration, sample_rate)
    square_wave = square_wave_signal(frequency, duration, sample_rate)
    sawtooth_wave = sawtooth_wave_signal(frequency, duration, sample_rate)
    triangle_wave = triangle_wave_signal(frequency, duration, sample_rate)

    fig, axs = plt.subplots(4, 1, figsize=(10, 8))
    axs[0].plot(sinusoidal)
    axs[0].set_title('Sinusoidal Signal')
    axs[1].plot(square_wave) 
    axs[1].set_title('Square Wave Signal')
    axs[2].plot(sawtooth_wave)
    axs[2].set_title('Sawtooth Wave Signal')
    axs[3].plot(triangle_wave)
    axs[3].set_title('Triangle Wave Signal')
    plt.tight_layout()
    plt.show()    