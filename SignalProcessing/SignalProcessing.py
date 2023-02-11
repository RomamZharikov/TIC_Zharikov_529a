from numpy.random import normal
from numpy import abs, arange
from scipy import fft
from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt


class SignalProcessing:
    def __init__(self, signal_length, frequency, frequency_max):
        self.__mean_distribution = 0
        self.__standard_deviation = 10
        self.__signal_length = signal_length
        self.__frequency = frequency
        self.__frequency_max = frequency_max

    def __signal_generation(self):
        return normal(self.__mean_distribution, self.__standard_deviation, self.__signal_length)

    def __time_ox(self):
        return arange(self.__signal_length) / self.__frequency

    def __parameters_filter(self):
        w = self.__frequency_max / (self.__frequency / 2)
        return butter(3, w, 'low', output='sos')

    def __filter(self):
        return sosfiltfilt(self.__parameters_filter(), self.__signal_generation())

    def __graphics(self, x_lab, x, y_lab, y, type_graphics):
        title = f"{type_graphics} з максимальною частотою F_max = {self.__frequency_max} Гц"
        fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
        ax.plot(x, y, linewidth=1)
        ax.set_xlabel(x_lab, fontsize=14)
        ax.set_ylabel(y_lab, fontsize=14)
        plt.title(title, fontsize=14)
        fig.savefig(fname=f"./figures/{title}", dpi=600)

    def __signal_spectrum(self):
        x = abs(fft.fftshift(fft.fft(self.__filter())))
        y = fft.fftshift(fft.fftfreq(self.__signal_length, 1 / self.__signal_length))
        return y, x

    def result(self):
        self.__graphics(x_lab="Час (секунди)", y_lab="Амплітуда сигналу", x=self.__time_ox(), y=self.__filter(),
                        type_graphics="Сигнал")
        self.__graphics(x_lab="Частота (Гц)", y_lab="Амплітуда спектру", x=self.__signal_spectrum()[0],
                        y=self.__signal_spectrum()[1], type_graphics="Спектр сигналу")


if __name__ == "__main__":
    a = SignalProcessing(500, 1000, 15)
    a.result()
