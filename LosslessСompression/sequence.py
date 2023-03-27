import random
import string
import collections
import math

from matplotlib import pyplot as plt


def origin_1(N_sequence, student_id):
    seq = ['0'] * (N_sequence - student_id) + ['1'] * student_id
    random.shuffle(seq)
    return ''.join(seq)


def origin_2(N_sequence, surname):
    seq = list(surname) + ['0'] * (N_sequence - len(surname))
    return ''.join(seq)


def origin_3(N_sequence, surname):
    list1 = list(surname)
    N1 = len(list1)
    list0 = ['0'] * (N_sequence - N1)
    seq = list1 + list0
    random.shuffle(seq)
    return ''.join(seq)


def origin_4(N_sequence, surname, group_number):
    letters = list(surname + group_number[:3])
    n_letters = len(letters)
    n_repeats = N_sequence // n_letters
    remainder = N_sequence % n_letters
    sequence_list = letters * n_repeats
    sequence_list += letters[:remainder]
    return ''.join(map(str, sequence_list))


def origin_5(N_sequence, surname, group_number, pi):
    letters = [*list(surname[:2]), *list(group_number[:3])]
    for i in letters:
        if 100 / len(letters) == 20:
            letters100 = [i for i in letters for j in range(20)]
        else:
            letters100 = [random.choices(letters, weights=[pi] * len(letters))[0] for i in range(N_sequence)]
        random.shuffle(letters100)
    return "".join(letters100)


def origin_6(N_sequence, surname, group_number, p_letters, p_digits):
    letters = list(surname[:2])
    digits = list(group_number[:3])
    n_letters = int(p_letters * N_sequence)
    n_digits = int(p_digits * N_sequence)
    list_100 = []
    for i in range(n_letters):
        list_100.append(random.choice(letters))
    for i in range(n_digits):
        list_100.append(random.choice(digits))
    random.shuffle(list_100)
    return "".join(list_100)


def origin_7(N_sequence):
    elements = string.ascii_lowercase + string.digits
    list_100 = [random.choice(elements) for i in range(N_sequence)]
    return "".join(list_100)


def origin_8(N_sequence):
    list_100 = ['1' for i in range(N_sequence)]
    return ''.join(list_100)


def result_1(origins):
    result = []
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
            f.write(f"""Послідовність {origins.index(sequence) + 1}
            Послідовність: {sequence}
            Розмір алфавіту: {n_sequence}
            Ймовірність появи символів: {''.join(chance)}
            Ймовірність розподілу символів: {is_equal_probability}
            Ентропія: {round(entropy, 2)}
            Надмірність джерела: {round(redundancy, 2)}\n\n""")
            result.append([len(counts), round(entropy, 2), round(redundancy, 2), is_equal_probability])
    f.close()
    save_img(result)
    save_txt(origins)


def save_img(result):
    headers = ['Розмір алфавіту', 'Ентропія', 'Надмірність', 'Ймовірність']
    fig, ax = plt.subplots(figsize=(14 / 1.54, 8 / 1.54))
    row = ['Послідовність 1', 'Послідовність 2', 'Послідовність 3', 'Послідовність 4', 'Послідовність 5',
           'Послідовність 6', 'Послідовність 7', 'Послідовність 8']
    ax.axis('off')
    table = ax.table(cellText=result, colLabels=headers, rowLabels=row,
                     loc='center', cellLoc='center')
    table.set_fontsize(14)
    table.scale(0.8, 2)
    fig.savefig(f"./results/Характеристики сформованих послідовностей.png", dpi=600)


def save_txt(origins):
    with open('./results/sequence.txt', 'w', encoding='utf-8') as f:
        for i in origins:
            f.write(i + "\n")
    f.close()


if __name__ == "__main__":
    num, index, surname, group = 100, 6, "Zharikov".lower(), "529a"
    origin_1 = origin_1(num, index)
    origin_2 = origin_2(num, surname)
    origin_3 = origin_3(num, surname)
    origin_4 = origin_4(num, surname, group)
    origin_5 = origin_5(num, surname, group, 0.2)
    origin_6 = origin_6(num, surname, group, 0.7, 0.3)
    origin_7 = origin_7(num)
    origin_8 = origin_8(num)
    origins = [origin_1, origin_2, origin_3, origin_4, origin_5, origin_6, origin_7, origin_8]
    result_1(origins)
