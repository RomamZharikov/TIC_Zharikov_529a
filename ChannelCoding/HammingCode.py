import random

CHUNK_LENGTH = 8
assert not CHUNK_LENGTH % 8, 'Довжина блоку має бути кратна 8'
CHECK_BITS = [i for i in range(1, CHUNK_LENGTH + 1) if not i & (i - 1)]


def getCharsToBin(chars):
    assert not len(chars) * 8 % CHUNK_LENGTH, 'Довжина кодових даних повинна бути кратною довжині блоку кодування'
    return ''.join([bin(ord(c))[2:].zfill(8) for c in chars])


def getChunkIterator(text_bin, chunk_size=CHUNK_LENGTH):
    for i in range(len(text_bin)):
        if not i % chunk_size:
            yield text_bin[i:i + chunk_size]


def getCheckBitsData(value_bin):
    check_bits_count_map = {k: 0 for k in CHECK_BITS}
    for index, value in enumerate(value_bin, 1):
        if int(value):
            bin_char_list = list(bin(index)[2:].zfill(8))
            bin_char_list.reverse()
            for degree in [2 ** int(i) for i, value in enumerate(bin_char_list) if int(value)]:
                check_bits_count_map[degree] += 1
    check_bits_value_map = {}
    for check_bit, count in check_bits_count_map.items():
        check_bits_value_map[check_bit] = 0 if not count % 2 else 1
    return check_bits_value_map


def getSetEmptyCheckBits(value_bin):
    for bit in CHECK_BITS:
        value_bin = value_bin[:bit - 1] + '0' + value_bin[bit - 1:]
    return value_bin


def getSetCheckBits(value_bin):
    value_bin = getSetEmptyCheckBits(value_bin)
    check_bits_data = getCheckBitsData(value_bin)
    for check_bit, bit_value in check_bits_data.items():
        value_bin = '{0}{1}{2}'.format(value_bin[:check_bit - 1], bit_value, value_bin[check_bit:])
    return value_bin


def getCheckBits(value_bin):
    check_bits = {}
    for index, value in enumerate(value_bin, 1):
        if index in CHECK_BITS:
            check_bits[index] = int(value)
    return check_bits


def getExcludeCheckBits(value_bin):
    clean_value_bin = ''
    for index, char_bin in enumerate(list(value_bin), 1):
        if index not in CHECK_BITS:
            clean_value_bin += char_bin
    return clean_value_bin


def getSetErrors(encoded):
    result = ''
    for chunk in getChunkIterator(encoded, CHUNK_LENGTH + len(CHECK_BITS)):
        num_bit = random.randint(1, len(chunk))
        chunk = '{0}{1}{2}'.format(chunk[:num_bit - 1], int(chunk[num_bit - 1]) ^ 1, chunk[num_bit:])
        result += chunk
    return result


def getCheckAndFixError(encoded_chunk):
    check_bits_encoded = getCheckBits(encoded_chunk)
    check_item = getExcludeCheckBits(encoded_chunk)
    check_item = getSetCheckBits(check_item)
    check_bits = getCheckBits(check_item)
    if check_bits_encoded != check_bits:
        invalid_bits = []
        for check_bit_encoded, value in check_bits_encoded.items():
            if check_bits[check_bit_encoded] != value:
                invalid_bits.append(check_bit_encoded)
        num_bit = sum(invalid_bits)
        encoded_chunk = '{0}{1}{2}'.format(encoded_chunk[:num_bit - 1],
                                           int(encoded_chunk[num_bit - 1]) ^ 1, encoded_chunk[num_bit:])
    return encoded_chunk


def getDiffIndexList(value_bin1, value_bin2):
    diff_index_list = []
    for index, char_bin_items in enumerate(zip(list(value_bin1), list(value_bin2)), 1):
        if char_bin_items[0] != char_bin_items[1]:
            diff_index_list.append(index)
    return diff_index_list


def encode(source):
    text_bin = getCharsToBin(source)
    result = ''
    for chunk_bin in getChunkIterator(text_bin):
        chunk_bin = getSetCheckBits(chunk_bin)
        result += chunk_bin
    return text_bin, result


def decode(encoded, fix_errors=True):
    decoded_value = ''
    fixed_encoded_list = []
    for encoded_chunk in getChunkIterator(encoded, CHUNK_LENGTH + len(CHECK_BITS)):
        if fix_errors:
            encoded_chunk = getCheckAndFixError(encoded_chunk)
        fixed_encoded_list.append(encoded_chunk)
    clean_chunk_list = []
    for encoded_chunk in fixed_encoded_list:
        encoded_chunk = getExcludeCheckBits(encoded_chunk)
        clean_chunk_list.append(encoded_chunk)
    for clean_chunk in clean_chunk_list:
        for clean_char in [clean_chunk[i:i + 8] for i in range(len(clean_chunk)) if not i % 8]:
            decoded_value += chr(int(clean_char, 2))
    return decoded_value


if __name__ == '__main__':
    with open("result_hamming.txt", "w", encoding="utf-8") as f:
        with open("../LosslessСompression/results/sequence.txt", "r", encoding="utf-8") as file:
            original_sequences = [line.strip() for line in file.readlines()]
        original_sequences = [seq.strip("[]").strip("'") for seq in original_sequences]
        for sequence in original_sequences:
            source = sequence[:10]
            source_bin, encoded = encode(source)
            decoded = decode(encoded)
            encoded_with_error = getSetErrors(encoded)
            diff_index_list = getDiffIndexList(encoded, encoded_with_error)
            decoded_with_error = decode(encoded_with_error, fix_errors=False)
            decoded_without_error = decode(encoded_with_error)
            f.write(f"""{'/' * 70}
        Початкова послідовність №{original_sequences.index(sequence)+1} (в байтах): {source}
Початкова послідовність (в бітах): {source_bin}
Довжина початкової послідовності (в бітах): {str(len(source_bin))}
Довжина блоку кодування: {str(CHUNK_LENGTH)}
Позиції контрольних бітів: {str(CHECK_BITS)}
Відносна надмірність коду: {str(len(CHECK_BITS) / CHUNK_LENGTH)}
{'-' * 15} Кодування {'-' * 15}
Закодовані дані: {encoded}
Довжина закодованих даних: {str(len(encoded))}
{'-' * 15} Декодування {'-' * 13}
Декодовані дані: {decoded}
Довжина декодованих даних (в бітах): {str(len(decoded) * 8)}
{'-' * 15} Внесення помилок {'-' * 8}
Закодовані дані з помилками: {encoded_with_error}
Кількість помилок: {str(len(diff_index_list))}
Позиції помилок: {str(diff_index_list)}
Декодовані дані з помилками: {decoded_with_error}
{'-' * 15} Виправлення помилок {'-' * 5}
Декодовані дані без виправлення помилки: {decoded_without_error}
{'/' * 70}\n\n""")





