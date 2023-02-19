from numpy.random import normal
from numpy import abs, arange, zeros, array, var
from scipy import fft
from scipy.signal import butter, sosfiltfilt
import matplotlib.pyplot as plt


class Filter:
    def __init__(self, frequency_max, frequency, signal):
        self.__frequency_max = frequency_max
        self.__frequency = frequency
        self.__signal = signal

    def __parameters_filter(self):
        w = self.__frequency_max / (self.__frequency / 2)
        return butter(3, w, 'low', output='sos')

    def filter(self):
        return sosfiltfilt(self.__parameters_filter(), self.__signal)

    def recovery_filter(self, signal: list):
        recovery_signal = []
        for i in signal:
            recovery_signal += [list(sosfiltfilt(self.__parameters_filter(), i))]
        return recovery_signal


class SignalProcessing:
    def __init__(self, signal_length, frequency, frequency_max, frequency_filter):
        self.__mean_distribution = 0
        self.__standard_deviation = 10
        self.__frequency_filter = frequency_filter
        self.__signal_length = signal_length
        self.__dt = [2, 4, 8, 16]
        self.__signal = normal(self.__mean_distribution, self.__standard_deviation, self.__signal_length)
        self.__frequency = frequency
        self.__frequency_max = frequency_max
        self.__low_pass_filter = Filter(frequency_max, frequency, self.__signal).filter()
        self.__recovery_filter = Filter(frequency_filter, frequency, self.__signal)

    def __time_ox(self):
        return arange(self.__signal_length) / self.__frequency

    def __graphics(self, x_lab, x, y_lab, y, type_graphics: list):
        if type_graphics[1] == "сигнал":
            fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
            title = f"{type_graphics[0]} з максимальною частотою F_max = {self.__frequency_max} Гц"
            ax.plot(x, y, linewidth=1)
            ax.set_xlabel(x_lab, fontsize=14)
            ax.set_ylabel(y_lab, fontsize=14)
            plt.title(title, fontsize=14)
        elif type_graphics[1] == "дискретизація":
            fig, ax = plt.subplots(2, 2, figsize=(21 / 2.54, 14 / 2.54))
            title = f"{type_graphics[0]} з кроком дискредитації Dt = {self.__dt} "
            s = 0
            for i in range(0, 2):
                for j in range(0, 2):
                    ax[i][j].plot(x, y[s], linewidth=1)
                    s += 1
            fig.supxlabel(x_lab, fontsize=14)
            fig.supylabel(y_lab, fontsize=14)
            plt.suptitle(title, fontsize=14)
        elif type_graphics[1] == "дисперсії" or type_graphics[1] == "співвідношення сигнал-шум":
            fig, ax = plt.subplots(figsize=(21 / 2.54, 14 / 2.54))
            title = f"Залежність {type_graphics[1]} від кроку дискретизації"
            ax.plot(x, y, linewidth=1)
            ax.grid()
            ax.set_xlabel(x_lab, fontsize=14)
            ax.set_ylabel(y_lab, fontsize=14)
            plt.title(title, fontsize=14)
        else:
            return False
        fig.savefig(fname=f"./figures/{title}", dpi=600)

    def __signal_spectrum(self):
        x = abs(fft.fftshift(fft.fft(self.__low_pass_filter)))
        y = fft.fftshift(fft.fftfreq(self.__signal_length, 1 / self.__signal_length))
        return y, x

    def __discretization_signal(self):
        discrete_signals = []
        for Dt in self.__dt:
            discrete_signal = zeros(self.__signal_length)
            for i in range(0, round(self.__signal_length / Dt)):
                discrete_signal[i * Dt] = self.__recovery_filter.filter()[i * Dt]
            discrete_signals += [list(discrete_signal)]
        return discrete_signals

    def __discretization_spectrum(self):
        x = fft.fftshift(fft.fftfreq(self.__signal_length, 1 / self.__signal_length))
        discrete_spectrum = []
        for Dt in range(len(self.__dt)):
            y = abs(fft.fftshift(fft.fft(self.__discretization_signal()[Dt])))
            discrete_spectrum += [list(y)]
        return x, discrete_spectrum

    def __signal_recovery(self):
        return self.__recovery_filter.recovery_filter(self.__discretization_signal())

    def __signals(self, type_graphics):
        self.__graphics(x_lab="Час (секунди)", y_lab="Амплітуда сигналу", x=self.__time_ox(), y=self.__low_pass_filter,
                        type_graphics=["Сигнал", type_graphics])
        self.__graphics(x_lab="Частота (Гц)", y_lab="Амплітуда спектру", x=self.__signal_spectrum()[0],
                        y=self.__signal_spectrum()[1], type_graphics=["Спектр сигналу", type_graphics])

    def __signal_difference(self):
        signal_difference = [self.__signal]
        for i in self.__signal_recovery():
            signal_difference += [array(i) - self.__signal]
        return signal_difference

    def __signal_dispersion(self):
        signal_dispersion = []
        for i in self.__signal_difference():
            signal_dispersion += [var(i)]
        return signal_dispersion

    def __signal_to_noise(self):
        signal_to_noise = []
        signal_dispersion = self.__signal_dispersion()
        for i in signal_dispersion:
            signal_to_noise += [(signal_dispersion[0] - i)]
        return signal_to_noise

    def __ratios(self):
        self.__graphics(x_lab="Крок дискретизації", y_lab="Дисперсія", x=self.__dt,
                        y=self.__signal_dispersion()[1:], type_graphics=[None, "дисперсії"])
        self.__graphics(x_lab="Крок дискретизації", y_lab="CCШ", x=self.__dt,
                        y=self.__signal_to_noise()[1:], type_graphics=[None, "співвідношення сигнал-шум"])

    def __discretization(self, type_graphics):
        self.__graphics(x_lab="Час (секунди)", y_lab="Амплітуда сигналу", x=self.__time_ox(),
                        y=self.__discretization_signal(), type_graphics=["Сигнал", type_graphics])
        x, y = self.__discretization_spectrum()
        self.__graphics(x_lab="Частота (Гц)", y_lab="Амплітуда спектру", x=x,
                        y=y, type_graphics=["Спектр сигналу", type_graphics])
        self.__graphics(x_lab="Час (секунди)", y_lab="Амплітуда сигналу", x=self.__time_ox(),
                        y=self.__signal_recovery(), type_graphics=["Відновлені аналогові сигнали", type_graphics])

    def result(self, type_graphics: str):
        type_graphics = type_graphics.lower().strip()
        if type_graphics == "сигнал":
            self.__signals("сигнал")
        elif type_graphics == "дискретизація":
            self.__discretization("дискретизація")
        elif type_graphics == "співвідношення":
            self.__ratios()
        elif type_graphics == "всі":
            self.__signals("сигнал")
            self.__discretization("дискретизація")
            self.__ratios()


if __name__ == "__main__":
    a = SignalProcessing(500, 1000, 15, 22)
    a.result(type_graphics="всі")
