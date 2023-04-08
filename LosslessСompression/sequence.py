import random
import string
import collections
import math
from matplotlib import pyplot as plt


class Generated_sequences:
    def __init__(self, num, index, surname, group, pi, p_letters, p_digits):
        self.__origins = []
        self.__num = num
        self.__index = index
        self.__surname = surname
        self.__group = group
        self.__pi = pi
        self.__p_letters = p_letters
        self.__p_digits = p_digits

    def __origin_1(self):
        seq = ['0'] * (self.__num - self.__index) + ['1'] * self.__index
        random.shuffle(seq)
        return ''.join(seq)

    def __origin_2(self):
        seq = list(self.__surname) + ['0'] * (self.__num - len(self.__surname))
        return ''.join(seq)

    def __origin_3(self):
        list1 = list(self.__surname)
        n1 = len(list1)
        list0 = ['0'] * (self.__num - n1)
        seq = list1 + list0
        random.shuffle(seq)
        return ''.join(seq)

    def __origin_4(self):
        letters = list(self.__surname + self.__group[:3])
        n_letters = len(letters)
        n_repeats = self.__num // n_letters
        remainder = self.__num % n_letters
        sequence_list = letters * n_repeats
        sequence_list += letters[:remainder]
        return ''.join(map(str, sequence_list))

    def __origin_5(self):
        letters = [*list(self.__surname[:2]), *list(self.__group[:3])]
        for i in letters:
            if 100 / len(letters) == 20:
                letters100 = [i for i in letters for j in range(20)]
            else:
                letters100 = [random.choices(letters, weights=[self.__pi] * len(letters))[0] for i in range(self.__num)]
            random.shuffle(letters100)
        return "".join(letters100)

    def __origin_6(self):
        letters = list(self.__surname[:2])
        digits = list(self.__group[:3])
        n_letters = int(self.__p_letters * self.__num)
        n_digits = int(self.__p_digits * self.__num)
        list_100 = []
        for i in range(n_letters):
            list_100.append(random.choice(letters))
        for i in range(n_digits):
            list_100.append(random.choice(digits))
        random.shuffle(list_100)
        return "".join(list_100)

    def __origin_7(self):
        elements = string.ascii_lowercase + string.digits
        list_100 = [random.choice(elements) for i in range(self.__num)]
        return "".join(list_100)

    def __origin_8(self):
        list_100 = ['1' for i in range(self.__num)]
        return ''.join(list_100)

    def __result_1(self):
        result = []
        origins = self.__origins
        with open('./results/results_sequence.txt', 'w', encoding='utf-8') as f:
            for sequence in origins:
                counts = collections.Counter(sequence)
                n_sequence = len(sequence)
                probability = {symbol: count / n_sequence for symbol, count in counts.items()}
                mean_probability = sum(probability.values()) / len(probability)
                is_equal_probability = all(abs(prob - mean_probability) < 0.05 * mean_probability for prob in
                                           probability.values())
                if is_equal_probability:
                    is_equal_probability = "рівна"
                elif not is_equal_probability:
                    is_equal_probability = "нерівна"
                entropy = -sum(probability[symbol] * math.log(probability[symbol], 2) for symbol in probability)
                if len(counts) > 1:
                    redundancy = 1 - entropy / math.log2(len(counts))
                else:
                    redundancy = 1
                chance = []
                for i, j in probability.items():
                    chance += [f"{i}={j}, "]
                if entropy == 0:
                    entropy = 0
                f.write(f"""Послідовність {self.__origins.index(sequence) + 1}
                Послідовність: {sequence}
                Розмір алфавіту: {n_sequence}
                Ймовірність появи символів: {''.join(chance)}
                Ймовірність розподілу символів: {is_equal_probability}
                Ентропія: {round(entropy, 2)}
                Надмірність джерела: {round(redundancy, 2)}\n\n""")
                result.append([len(counts), round(entropy, 2), round(redundancy, 2), is_equal_probability])
        f.close()
        return result

    def __save_img(self):
        headers = ['Розмір алфавіту', 'Ентропія', 'Надмірність', 'Ймовірність']
        fig, ax = plt.subplots(figsize=(14 / 1.54, 8 / 1.54))
        row = ['Послідовність 1', 'Послідовність 2', 'Послідовність 3', 'Послідовність 4', 'Послідовність 5',
               'Послідовність 6', 'Послідовність 7', 'Послідовність 8']
        ax.axis('off')
        table = ax.table(cellText=self.__result_1(), colLabels=headers, rowLabels=row,
                         loc='center', cellLoc='center')
        table.set_fontsize(14)
        table.scale(0.8, 2)
        fig.savefig(f"./results/Характеристики сформованих послідовностей.png", dpi=600)

    def __save_txt(self):
        with open('./results/sequence.txt', 'w', encoding='utf-8') as f:
            for i in self.__origins:
                f.write(i + "\n")
        f.close()

    def main(self):
        self.__origins.clear()
        origin_1 = self.__origin_1()
        origin_2 = self.__origin_2()
        origin_3 = self.__origin_3()
        origin_4 = self.__origin_4()
        origin_5 = self.__origin_5()
        origin_6 = self.__origin_6()
        origin_7 = self.__origin_7()
        origin_8 = self.__origin_8()
        self.__origins += [origin_1, origin_2, origin_3, origin_4, origin_5, origin_6, origin_7, origin_8]
        self.__result_1()
        self.__save_img()
        self.__save_txt()


class RLE_LZW:
    def __init__(self):
        self.__origins = []
        with open('./results/sequence.txt', 'r', encoding='utf-8') as f:
            for i in f:
                self.__origins.append(i[:-1])

    def __encode_sequence_rle(self, sequence):
        result = []
        count = 1
        for i, item in enumerate(sequence):
            if i == 0:
                continue
            if item == sequence[i - 1]:
                count += 1
            else:
                result.append((sequence[i - 1], count))
                count = 1
        result.append((sequence[len(sequence) - 1], count))
        encoded = []
        for i, item in enumerate(result):
            encoded.append(f"{item[1]}{item[0]}")
        return encoded

    def __encode_sequence_lzw(self, sequence):
        dictionary_encode = {}
        for i in range(65536):
            dictionary_encode[chr(i)] = i
        current = ""
        size = 0
        result = []
        for char in sequence:
            new_str = current + char
            if new_str in dictionary_encode:
                current = new_str
            else:
                result.append(dictionary_encode[current])
                dictionary_encode[new_str] = len(dictionary_encode)
                element_bits = 16 if dictionary_encode[current] < 65536 else math.ceil(
                    math.log2(len(dictionary_encode)))
                with open("./results/results_rle_lzw.txt", "a", encoding='utf-8') as output_file:
                    output_file.write(
                        f"Code: {dictionary_encode[current]}, Element: {current}, bits: {element_bits}\n")
                size += element_bits
                current = char
        result.append(dictionary_encode[current])
        last_bits = 16 if dictionary_encode[current] < 65536 else math.ceil(math.log2(len(dictionary_encode)))
        with open("./results/results_rle_lzw.txt", "a", encoding='utf-8') as output_file:
            output_file.write("____________________________________________\n")
            output_file.write(f"Закодована LZW послідовність: {''.join(map(str, result))}\n")
            output_file.write(f"Розмір закодованої LZW послідовності: {size + last_bits} bits\n")
        return result, size + last_bits

    def __decode_sequence_rle(self, sequence):
        result = []
        for item in sequence:
            result.append("".join([int(item[:-1]) * item[-1]]))
        return result

    def __decode_sequence_lzw(self, sequence):
        dictionary_decode = {}
        for i in range(65536):
            dictionary_decode[i] = chr(i)
        decoded_message = dictionary_decode[sequence[0]]
        current_sequence = decoded_message
        for code in sequence[1:]:
            if code in dictionary_decode:
                sequence = dictionary_decode[code]
            else:
                sequence = current_sequence + current_sequence[0]
            decoded_message += sequence
            dictionary_decode[len(dictionary_decode)] = current_sequence + sequence[0]
            current_sequence = sequence
        return decoded_message

    def __entropy(self, sequence):
        counts = collections.Counter(sequence)
        probability = {symbol: count / len(sequence) for symbol, count in counts.items()}
        entropy = round(-sum(probability[symbol] * math.log(probability[symbol], 2) for symbol in probability), 2)
        if entropy == 0:
            entropy = 0
        return entropy

    def plot(self, results):
        fig, ax = plt.subplots(figsize=(14 / 1.54, 8 / 1.54))
        headers = ['Ентропія', 'КС RLE', 'КС LZW']
        row = ['Послідовність 1', 'Послідовність 2', 'Послідовність 3', 'Послідовність 4', 'Послідовність 5',
               'Послідовність 6', 'Послідовність 7', 'Послідовність 8']
        ax.axis('off')
        table = ax.table(cellText=results, colLabels=headers, rowLabels=row,
                         loc='center', cellLoc='center')
        table.set_fontsize(14)
        table.scale(0.8, 2)
        fig.savefig(f"./results/Результати стиснення методами RLE та LZW.png", dpi=600)

    def main(self):
        results = []
        for sequence in self.__origins:
            with open('./results/results_rle_lzw.txt', 'a', encoding='utf-8') as f:
                f.write("//////////////////////////////////////////////////////////////////////\n")
                f.write(f"Оригінальна послідовність: {sequence}\n")
                f.write(f"Розмір оригінальної послідовності: {len(sequence) * 16} bits\n")
                entropy = self.__entropy(sequence)
                f.write(f"Ентропія: {entropy}\n\n")
                encoded_rle = self.__encode_sequence_rle(sequence)
                encoded_rle_result = "".join(encoded_rle)
                decoded_rle = self.__decode_sequence_rle(encoded_rle)
                decoded_rle_result = "".join(decoded_rle)
                compression_ratio_rle = round((len(sequence) / (len(encoded_rle_result))), 2)
                if compression_ratio_rle < 1:
                    compression_ratio_rle = '-'
                f.write("_______________ RLE кодування _______________\n")
                f.write(f"Закодована послідовність: {encoded_rle_result}\n")
                f.write(f"Розмір закодованої послідовності: {len(encoded_rle_result) * 16} bits\n")
                f.write(f"Коефіцієнт стиснення RLE: {compression_ratio_rle}\n")
                f.write(f"Декодована RLE послідовність: {decoded_rle_result}\n")
                f.write(f"Розмір декодованої RLE послідовності: {len(decoded_rle_result) * 16} bits\n\n")
                f.write("_______________ LZW кодування _______________\n")
                f.write("____________ Поетапне кодування ____________\n")
            encoded_lzw, size_lzw = self.__encode_sequence_lzw(sequence)
            decoded_lzw = self.__decode_sequence_lzw(encoded_lzw)
            compression_ratio_lzw = round(len(sequence) * 16 / (size_lzw), 2)
            with open('./results/results_rle_lzw.txt', 'a', encoding='utf-8') as f:
                f.write(f"Коефіцієнт стиснення LZW: {compression_ratio_lzw}\n")
                f.write(f"Декодована LZW послідовність: {decoded_lzw}\n")
                f.write(f"Розмір декодованої LZW послідовності: {len(decoded_lzw) * 16} bits\n\n")
            results.append([round(entropy, 2), compression_ratio_rle, compression_ratio_lzw])
        self.plot(results)


class AC_CH:
    def __init__(self):
        self.__origins = []
        with open('./results/sequence.txt', 'r', encoding='utf-8') as f:
            for i in f:
                self.__origins.append(i[:10])

    def origin_sequence(self, sequence):
        sequence_length = len(sequence)
        unique_chars = set(sequence)
        sequence_alphabet_size = len(unique_chars)
        counts = collections.Counter(sequence)
        n_sequence = 10
        probability = {symbol: count / n_sequence for symbol, count in counts.items()}
        entropy = -sum(p * math.log2(p) for p in probability.values())
        if entropy == 0:
            entropy = 0
        else:
            entropy = round(entropy, 4)
        return entropy, unique_chars, probability, sequence_alphabet_size, sequence_length

    def float_bin(self, point, size_cod):
        binary_code = ""
        for x in range(size_cod):
            point = point * 2
            if point > 1:
                binary_code = binary_code + str(1)
                x = int(point)
                point = point - x
            elif point < 1:
                binary_code = binary_code + str(0)
            else:
                binary_code = binary_code + str(1)
        return binary_code

    def encode_ac(self, unique_chars, probabilitys, alphabet_size, sequence):
        alphabet = list(unique_chars)
        probability = [probabilitys[symbol] for symbol in alphabet]
        unity = []
        probability_range = 0.0
        for i in range(alphabet_size):
            l = probability_range
            probability_range = probability_range + probability[i]
            u = probability_range
            unity.append([alphabet[i], l, u])
        for i in range(len(sequence) - 1):
            for j in range(len(unity)):
                if sequence[i] == unity[j][0]:
                    probability_low = unity[j][1]
                    probability_high = unity[j][2]
                    diff = probability_high - probability_low
                    for k in range(len(unity)):
                        unity[k][1] = probability_low
                        unity[k][2] = probability[k] * diff + probability_low
                        probability_low = unity[k][2]
                    break
        low = 0
        high = 0
        for i in range(len(unity)):
            if unity[i][0] == sequence[-1]:
                low = unity[i][1]
                high = unity[i][2]
        point = (low + high) / 2
        size_cod = math.ceil(math.log((1 / (high - low)), 2) + 1)
        bin_code = self.float_bin(point, size_cod)
        return [point, alphabet_size, alphabet, probability], bin_code

    def decode_ac(self, encoded_data_ac, sequence_length):
        point, alphabet_size, alphabet, probability = encoded_data_ac
        unity = []
        probability_range = 0.0
        for i in range(alphabet_size):
            l = probability_range
            probability_range = probability_range + probability[i]
            u = probability_range
            unity.append([alphabet[i], l, u])
        decoded_sequence = ""
        for i in range(sequence_length):
            for j in range(len(unity)):
                if unity[j][1] < point < unity[j][2]:
                    prob_low = unity[j][1]
                    prob_high = unity[j][2]
                    diff = prob_high - prob_low
                    decoded_sequence = decoded_sequence + unity[j][0]
                    for k in range(len(unity)):
                        unity[k][1] = prob_low
                        unity[k][2] = probability[k] * diff + prob_low
                        prob_low = unity[k][2]
                    break
        return decoded_sequence

    def encode_ch(self, unique_chars, probabilities, sequence):
        alphabet = list(unique_chars)
        probability = [probabilities[symbol] for symbol in alphabet]
        final = []
        for i in range(len(alphabet)):
            final.append([alphabet[i], probability[i]])
        final.sort(key=lambda x: x[1])
        tree = []
        if len(set(probability)) == 1:
            symbol_code = []
            for i in range(len(alphabet)):
                code = "1" * i + "0"
                symbol_code.append([alphabet[i], code])
            encode = "".join([symbol_code[alphabet.index(c)][1] for c in sequence])
        else:
            while len(final) > 1:
                left = final.pop(0)
                right = final.pop(0)
                tot = left[1] + right[1]
                tree.append([left[0], right[0]])
                final.append([left[0] + right[0], tot])
                final.sort(key=lambda x: x[1])
            symbol_code = []
            tree.reverse()
            alphabet.sort()
            for i in range(len(alphabet)):
                code = ""
                for j in range(len(tree)):
                    if alphabet[i] in tree[j][0]:
                        code = code + '0'
                        if alphabet[i] == tree[j][0]:
                            break
                    else:
                        code = code + '1'
                        if alphabet[i] == tree[j][1]:
                            break
                symbol_code.append([alphabet[i], code])
            encode = ""
            for c in sequence:
                encode += [symbol_code[i][1] for i in range(len(alphabet)) if
                           symbol_code[i][0] == c][0]
        return [encode, symbol_code], encode

    def decode_ch(self, encoded_sequence):
        encode = list(encoded_sequence[0])
        symbol_code = encoded_sequence[1]
        count = 0
        flag = 0
        sequence = ""
        for i in range(len(encode)):
            for j in range(len(symbol_code)):
                if encode[i] == symbol_code[j][1]:
                    sequence = sequence + str(symbol_code[j][0])
                    flag = 1
            if flag == 1:
                flag = 0
            else:
                count = count + 1
                if count == len(encode):
                    break
                else:
                    encode.insert(i + 1, str(encode[i] + encode[i + 1]))
                encode.pop(i + 2)
        return sequence

    def plot(self, results):
        fig, ax = plt.subplots(figsize=(14 / 1.54, 8 / 1.54))
        headers = ['Ентропія', 'bps AC', 'bps HC']
        row = ['Послідовність 1', 'Послідовність 2', 'Послідовність 3', 'Послідовність 4', 'Послідовність 5',
               'Послідовність 6', 'Послідовність 7', 'Послідовність 8']
        ax.axis('off')
        table = ax.table(cellText=results, colLabels=headers, rowLabels=row,
                         loc='center', cellLoc='center')
        table.set_fontsize(14)
        table.scale(0.8, 2)
        fig.savefig(f"./results/Результати стиснення методами AC та CH.png", dpi=600)

    def main(self):
        result = []
        with open('./results/results_AC_CH.txt', 'w', encoding='utf-8') as f:
            pass
        for sequence in self.__origins:
            entropy, unique_chars, probability, alphabet_size, sequence_length = self.origin_sequence(sequence)
            with open('./results/results_AC_CH.txt', 'a', encoding='utf-8') as f:
                f.write(f"""{'/' * 70}
    Послідовність {self.__origins.index(sequence) + 1}
Оригінальна послідовність: {sequence}
Ентропія: {entropy}""")
            encoded_data_ac, encoded_sequence_ac = self.encode_ac(unique_chars,
                                                                  probability, alphabet_size, sequence)
            bps_ac = len(encoded_sequence_ac) / sequence_length
            decoded_sequence_ac = self.decode_ac(encoded_data_ac, sequence_length)
            encoded_data_hc, encoded_sequence_hc = self.encode_ch(unique_chars, probability, sequence)
            bps_hc = len(encoded_sequence_hc) / sequence_length
            decoded_sequence_hc = self.decode_ch(encoded_data_hc)
            with open('./results/results_AC_CH.txt', 'a', encoding='utf-8') as f:
                f.write(f"""\n\n{10 * "_"} Арифметичне кодування {10 * "_"}
Дані закодованої АС послідовності: {encoded_data_ac}
Закодована АС послідовність: {encoded_sequence_ac}
Значення bps при кодуванні АС: {bps_ac}
Декодована АС послідовність: {decoded_sequence_ac}\n\n""")
                f.write(f"""{10 * "_"} Кодування Хаффмана {10 * "_"}
Алфавіт     Код символу\n""")
                for i, j in encoded_data_hc[1]:
                    f.write(f"{i}           {j}\n")
                f.write(f"""Дані закодованої НС послідовності: {encoded_data_hc}
Закодована НС послідовність: {encoded_sequence_hc}
Значення bps при кодуванні НС: {bps_hc}
Декодована НС послідовність: {decoded_sequence_hc}\n\n""")
                result.append([round(entropy, 2), bps_ac, bps_hc])
        self.plot(result)


if __name__ == "__main__":
    num, index, surname, group, pi, p_letters, p_digits = 100, 6, "Zharikov".lower(), "529a", 0.2, 0.3, 0.7
    choice = "3"
    if choice == "1":
        Generated_sequences(num, index, surname, group, pi, p_letters, p_digits).main()
    elif choice == "2":
        with open('./results/results_rle_lzw.txt', 'w', encoding='utf-8') as f:
            pass
        RLE_LZW().main()
    elif choice == "3":
        AC_CH().main()
