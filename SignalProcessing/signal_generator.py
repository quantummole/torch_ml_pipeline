import numpy as np
import simpleaudio as sa

def sinusoidal_signal(frequency, sample_rate):
    assert frequency > 0, "Frequency must be positive"
    assert sample_rate > 0, "Sample rate must be positive"
    assert sample_rate % frequency == 0, "Sample rate must be a multiple of frequency"
    cycle_length = sample_rate / frequency
    t = np.linspace(0, 1/frequency, int(cycle_length), endpoint=False)
    return np.sin(2 * np.pi * frequency * t)

def square_wave_signal(frequency, sample_rate):
    return np.sign(sinusoidal_signal(frequency, sample_rate))

def triangle_wave_signal(frequency, sample_rate):
    signal =  np.cumsum(square_wave_signal(frequency, sample_rate))
    signal = signal - np.min(signal)  # Normalize to start from zero
    signal = 2 * (signal / np.max(signal)) - 1  # Scale to [-1, 1]
    return signal

def sawtooth_wave_signal(frequency, sample_rate):
    cycle_length = sample_rate / frequency
    signal = triangle_wave_signal(frequency, 2*sample_rate)
    return signal[:int(cycle_length)]

def phase_shifted_signal(signal, phase_shift):
    signal_augmented = np.tile(signal, 2)  # Ensure the signal is long enough for shifting
    assert -np.pi <= phase_shift < np.pi, "Phase shift must be in the range [0, 1)"
    phase_shift = phase_shift + np.pi
    shift_samples = int(len(signal)*phase_shift / (2 * np.pi))
    return np.roll(signal_augmented, shift_samples)[:len(signal)]

def extend_duration(signal, duration, sample_rate):
    cycle_length = len(signal)
    num_cycles = int(np.ceil(duration * sample_rate / cycle_length))
    return np.tile(signal, num_cycles)[:int(duration * sample_rate)]

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    frequency = 5  # Hz
    duration = 1   # seconds
    sample_rate = 1000  # samples per second

    sinusoidal = sinusoidal_signal(frequency, sample_rate)
    square_wave = square_wave_signal(frequency, sample_rate)
    sawtooth_wave = sawtooth_wave_signal(frequency, sample_rate)
    triangle_wave = triangle_wave_signal(frequency, sample_rate)

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

    fig, axs = plt.subplots(4, 1, figsize=(10, 8))
    axs[0].plot(extend_duration(phase_shifted_signal(sinusoidal, np.pi/4),1,sample_rate))
    axs[0].set_title('Sinusoidal Signal')
    axs[1].plot(phase_shifted_signal(square_wave, np.pi/4))
    axs[1].set_title('Square Wave Signal')
    axs[2].plot(phase_shifted_signal(sawtooth_wave, np.pi/4))
    axs[2].set_title('Sawtooth Wave Signal')
    axs[3].plot(phase_shifted_signal(triangle_wave, np.pi/4))
    axs[3].set_title('Triangle Wave Signal')
    plt.tight_layout()
    plt.show()    

