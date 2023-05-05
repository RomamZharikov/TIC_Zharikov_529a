import os
from huffman import HuffmanTree
import math
import numpy as np
from scipy import fftpack
from PIL import Image


def dct_2d(image):
    return fftpack.dct(fftpack.dct(image.T, norm='ortho').T, norm='ortho')


def quantize(block, component, num_table):
    q = load_quantization_table(component, num_table)
    return (block / q).round().astype(np.int32)


def load_quantization_table(component, num_table):
    if num_table == 1:
        if component == 'lum':
            q = np.array([[2, 2, 2, 2, 3, 4, 5, 6],
                          [2, 2, 2, 2, 3, 4, 5, 6],
                          [2, 2, 2, 2, 4, 5, 7, 9],
                          [2, 2, 2, 4, 5, 7, 9, 12],
                          [3, 3, 4, 5, 8, 10, 12, 12],
                          [4, 4, 5, 7, 10, 12, 12, 12],
                          [5, 5, 7, 9, 12, 12, 12, 12],
                          [6, 6, 9, 12, 12, 12, 12, 12]])
        elif component == 'chrom':
            q = np.array([[3, 3, 5, 9, 13, 15, 15, 15],
                          [3, 4, 6, 11, 14, 12, 12, 12],
                          [5, 6, 9, 14, 12, 12, 12, 12],
                          [9, 11, 14, 12, 12, 12, 12, 12],
                          [13, 14, 12, 12, 12, 12, 12, 12],
                          [15, 12, 12, 12, 12, 12, 12, 12],
                          [15, 12, 12, 12, 12, 12, 12, 12],
                          [15, 12, 12, 12, 12, 12, 12, 12]])
        else:
            raise ValueError(" компонент має бути 'lum' або 'chrom', але '{comp}' знайдено".format(comp=component))
    elif num_table == 2:
        if component == 'lum':
            q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
                          [12, 12, 14, 19, 26, 48, 60, 55],
                          [14, 13, 16, 24, 40, 57, 69, 56],
                          [14, 17, 22, 29, 51, 87, 80, 62],
                          [18, 22, 37, 56, 68, 109, 103, 77],
                          [24, 35, 55, 64, 81, 104, 113, 92],
                          [49, 64, 78, 87, 103, 121, 120, 101],
                          [72, 92, 95, 98, 112, 100, 103, 99]])
        elif component == 'chrom':
            q = np.array([[17, 18, 24, 47, 99, 99, 99, 99],
                          [18, 21, 26, 66, 99, 99, 99, 99],
                          [24, 26, 56, 99, 99, 99, 99, 99],
                          [47, 66, 99, 99, 99, 99, 99, 99],
                          [99, 99, 99, 99, 99, 99, 99, 99],
                          [99, 99, 99, 99, 99, 99, 99, 99],
                          [99, 99, 99, 99, 99, 99, 99, 99],
                          [99, 99, 99, 99, 99, 99, 99, 99]])
        else:
            raise ValueError(" компонент має бути 'lum' або 'chrom', але '{comp}' знайдено".format(comp=component))
    else:
        raise ValueError(" номер таблиці має бути '1' або '2', але '{comp}' знайдено".format(comp=num_table))
    return q


def zigzag_points(rows, cols):
    UP, DOWN, RIGHT, LEFT, UP_RIGHT, DOWN_LEFT = range(6)

    def move(direction, point):
        return {
            UP: lambda point: (point[0] - 1, point[1]),
            DOWN: lambda point: (point[0] + 1, point[1]),
            LEFT: lambda point: (point[0], point[1] - 1),
            RIGHT: lambda point: (point[0], point[1] + 1),
            UP_RIGHT: lambda point: move(UP, move(RIGHT, point)),
            DOWN_LEFT: lambda point: move(DOWN, move(LEFT, point))
        }[direction](point)

    def inbounds(point):
        return 0 <= point[0] < rows and 0 <= point[1] < cols

    point = (0, 0)
    move_up = True
    for i in range(rows * cols):
        yield point
        if move_up:
            if inbounds(move(UP_RIGHT, point)):
                point = move(UP_RIGHT, point)
            else:
                move_up = False
                if inbounds(move(RIGHT, point)):
                    point = move(RIGHT, point)
                else:
                    point = move(DOWN, point)
        else:
            if inbounds(move(DOWN_LEFT, point)):
                point = move(DOWN_LEFT, point)
            else:
                move_up = True
                if inbounds(move(DOWN, point)):
                    point = move(DOWN, point)
                else:
                    point = move(RIGHT, point)


def block_to_zigzag(block):
    return np.array([block[point] for point in zigzag_points(*block.shape)])


def bits_required(n):
    n = abs(n)
    result = 0
    while n > 0:
        n >>= 1
        result += 1
    return result


def flatten(lst):
    return [item for sublist in lst for item in sublist]


def run_length_encode(arr):
    last_nonzero = -1
    for i, elem in enumerate(arr):
        if elem != 0:
            last_nonzero = i
    symbols = []
    values = []
    run_length = 0
    for i, elem in enumerate(arr):
        if i > last_nonzero:
            symbols.append((0, 0))
            values.append(int_to_binstr(0))
            break
        elif elem == 0 and run_length < 15:
            run_length += 1
        else:
            size = bits_required(elem)
            symbols.append((run_length, size))
            values.append(int_to_binstr(elem))
            run_length = 0
    return symbols, values


def binstr_flip(binstr):
    if not set(binstr).issubset('01'):
        raise ValueError("binstr повинен мати лише '0' та '1'")
    return ''.join(map(lambda c: '0' if c == '1' else '1', binstr))


def uint_to_binstr(number, size):
    return bin(number)[2:][-size:].zfill(size)


def int_to_binstr(n):
    if n == 0:
        return ''
    binstr = bin(abs(n))[2:]
    return binstr if n > 0 else binstr_flip(binstr)


def write_to_file(filepath, dc, ac, blocks_count, tables):
    f = open(filepath, 'w')
    for table_name in ['dc_y', 'ac_y', 'dc_c', 'ac_c']:
        f.write(uint_to_binstr(len(tables[table_name]), 16))
        for key, value in tables[table_name].items():
            if table_name in {'dc_y', 'dc_c'}:
                f.write(uint_to_binstr(key, 4))
                f.write(uint_to_binstr(len(value), 4))
                f.write(value)
            else:
                f.write(uint_to_binstr(key[0], 4))
                f.write(uint_to_binstr(key[1], 4))
                f.write(uint_to_binstr(len(value), 8))
                f.write(value)
    f.write(uint_to_binstr(blocks_count, 32))
    for b in range(blocks_count):
        for c in range(3):
            category = bits_required(dc[b, c])
            symbols, values = run_length_encode(ac[b, :, c])
            dc_table = tables['dc_y'] if c == 0 else tables['dc_c']
            ac_table = tables['ac_y'] if c == 0 else tables['ac_c']
            f.write(dc_table[category])
            f.write(int_to_binstr(dc[b, c]))
            for i in range(len(symbols)):
                f.write(ac_table[tuple(symbols[i])])
                f.write(values[i])
    f.close()


def encode(input_name, output_name, table_num):
    input_file = f"./input_image/{input_name}.bmp"
    output_file = f"./Results/{output_name}.asf"
    image = Image.open(input_file)
    ycbcr = image.convert('YCbCr')
    npmat = np.array(ycbcr, dtype=np.uint8)
    rows, cols = npmat.shape[0], npmat.shape[1]
    if rows % 8 == cols % 8 == 0:
        blocks_count = rows // 8 * cols // 8
    else:
        raise ValueError("ширина і висота зображення мають бути кратними 8")
    dc = np.empty((blocks_count, 3), dtype=np.int32)
    ac = np.empty((blocks_count, 63, 3), dtype=np.int32)
    for i in range(0, rows, 8):
        for j in range(0, cols, 8):
            try:
                block_index += 1
            except NameError:
                block_index = 0
            for k in range(3):
                block = npmat[i:i + 8, j:j + 8, k] - 128
                dct_matrix = dct_2d(block)
                quant_matrix = quantize(dct_matrix, 'lum' if k == 0 else 'chrom', table_num)
                zigzag = block_to_zigzag(quant_matrix)
                dc[block_index, k] = zigzag[0]
                ac[block_index, :, k] = zigzag[1:]
    H_DC_Y = HuffmanTree(np.vectorize(bits_required)(dc[:, 0]))
    H_DC_C = HuffmanTree(np.vectorize(bits_required)(dc[:, 1:].flat))
    H_AC_Y = HuffmanTree(
        flatten(run_length_encode(ac[i, :, 0])[0]
                for i in range(blocks_count)))
    H_AC_C = HuffmanTree(
        flatten(run_length_encode(ac[i, :, j])[0]
                for i in range(blocks_count) for j in [1, 2]))
    tables = {'dc_y': H_DC_Y.value_to_bitstring_table(),
              'ac_y': H_AC_Y.value_to_bitstring_table(),
              'dc_c': H_DC_C.value_to_bitstring_table(),
              'ac_c': H_AC_C.value_to_bitstring_table()}
    write_to_file(f"{output_file[:-4]}_{input_name}_{table_num}{output_file[-4:]}", dc, ac, blocks_count, tables)


def len_vyhsdnogo(input_file, action="num"):
    size_vyhsdnogo = os.path.getsize(f"./input_image/{input_file}.bmp")
    if action == "write":
        with open("results_jpeg.txt", "a", encoding='utf-8') as file:
            print('Розмір вихідного файла: {} байт\n'.format(size_vyhsdnogo), file=file)
        return None
    elif action == "num":
        return size_vyhsdnogo
    else:
        raise ValueError("action повинен бути 'write' та 'num'")


class JPEGFileReader:
    TABLE_SIZE_BITS = 16
    BLOCKS_COUNT_BITS = 32
    DC_CODE_LENGTH_BITS = 4
    CATEGORY_BITS = 4
    AC_CODE_LENGTH_BITS = 8
    RUN_LENGTH_BITS = 4
    SIZE_BITS = 4

    def __init__(self, filepath):
        self.__file = open(filepath, 'r')

    def read_int(self, size):
        if size == 0:
            return 0
        bin_num = self.__read_str(size)
        if bin_num[0] == '1':
            return self.__int2(bin_num)
        else:
            return self.__int2(binstr_flip(bin_num)) * -1

    def read_dc_table(self):
        table = dict()
        table_size = self.__read_uint(self.TABLE_SIZE_BITS)
        for _ in range(table_size):
            category = self.__read_uint(self.CATEGORY_BITS)
            code_length = self.__read_uint(self.DC_CODE_LENGTH_BITS)
            code = self.__read_str(code_length)
            table[code] = category
        return table

    def read_ac_table(self):
        table = dict()
        table_size = self.__read_uint(self.TABLE_SIZE_BITS)
        for _ in range(table_size):
            run_length = self.__read_uint(self.RUN_LENGTH_BITS)
            size = self.__read_uint(self.SIZE_BITS)
            code_length = self.__read_uint(self.AC_CODE_LENGTH_BITS)
            code = self.__read_str(code_length)
            table[code] = (run_length, size)
        return table

    def read_blocks_count(self):
        return self.__read_uint(self.BLOCKS_COUNT_BITS)

    def read_huffman_code(self, table):
        prefix = ''
        while prefix not in table:
            prefix += self.__read_char()
        return table[prefix]

    def __read_uint(self, size):
        if size <= 0:
            raise ValueError("розмір повинен бути більшим за 0")
        return self.__int2(self.__read_str(size))

    def __read_str(self, length):
        return self.__file.read(length)

    def __read_char(self):
        return self.__read_str(1)

    def __int2(self, bin_num):
        return int(bin_num, 2)


def read_image_file(filepath):
    reader = JPEGFileReader(filepath)
    tables = dict()
    for table_name in ['dc_y', 'ac_y', 'dc_c', 'ac_c']:
        if 'dc' in table_name:
            tables[table_name] = reader.read_dc_table()
        else:
            tables[table_name] = reader.read_ac_table()
    blocks_count = reader.read_blocks_count()
    dc = np.empty((blocks_count, 3), dtype=np.int32)
    ac = np.empty((blocks_count, 63, 3), dtype=np.int32)
    for block_index in range(blocks_count):
        for component in range(3):
            dc_table = tables['dc_y'] if component == 0 else tables['dc_c']
            ac_table = tables['ac_y'] if component == 0 else tables['ac_c']
            category = reader.read_huffman_code(dc_table)
            dc[block_index, component] = reader.read_int(category)
            ac_coefficients = []
            while len(ac_coefficients) < 63:
                run_length, size = reader.read_huffman_code(ac_table)
                if (run_length, size) == (0, 0):
                    ac_coefficients += [0] * (63 - len(ac_coefficients))
                else:
                    ac_coefficients += [0] * run_length
                    value = reader.read_int(size)
                    ac_coefficients.append(value)
            ac[block_index, :, component] = np.array(ac_coefficients)
    return dc, ac, tables, blocks_count


def zigzag_to_block(zigzag):
    rows = cols = int(math.sqrt(len(zigzag)))
    if rows * cols != len(zigzag):
        raise ValueError("Довжина зіг-зага повинна бути ідеальним квадратом ")
    block = np.empty((rows, cols), np.int32)
    for i, point in enumerate(zigzag_points(rows, cols)):
        block[point] = zigzag[i]
    return block


def dequantize(block, component, num_table):
    q = load_quantization_table(component, num_table)
    return block * q


def idct_2d(image):
    return fftpack.idct(fftpack.idct(image.T, norm='ortho').T, norm='ortho')


def decoder(input_bmp, input_asf, num_table):
    dc, ac, tables, blocks_count = read_image_file(f"./Results/{input_asf}_{input_bmp}_{num_table}.asf")
    block_side = 8
    image_side = int(math.sqrt(blocks_count)) * block_side
    blocks_per_line = image_side // block_side
    npmat = np.empty((image_side, image_side, 3), dtype=np.uint8)
    for block_index in range(blocks_count):
        i = block_index // blocks_per_line * block_side
        j = block_index % blocks_per_line * block_side
        for c in range(3):
            zigzag = [dc[block_index, c]] + list(ac[block_index, :, c])
            quant_matrix = zigzag_to_block(zigzag)
            dct_matrix = dequantize(quant_matrix, 'lum' if c == 0 else 'chrom', num_table)
            block = idct_2d(dct_matrix)
            npmat[i:i + 8, j:j + 8, c] = block + 128
    image = Image.fromarray(npmat, 'YCbCr')
    image = image.convert('RGB')
    filename = f"./Results/JPEG_{num_table}_{input_bmp}.jpg"
    image.save(filename)
    size_jpeg = os.path.getsize(filename)
    width, height = image.size
    ratio = len_vyhsdnogo(input_bmp) / size_jpeg
    with open("results_jpeg.txt", "a", encoding='utf-8') as file:
        print(f'Розмір файла JPEG: {size_jpeg} байт \nРозмір зображення JPEG: {width}x{height}', file=file)
        print('Коефіцієнт стиснення= {:.2f}'.format(ratio), '\n', file=file)


def get_file_names(directory):
    files = os.listdir(directory)
    file_names = []
    for file in files:
        file_name = os.path.splitext(file)[0]
        file_names.append(file_name)
    return file_names


if __name__ == "__main__":
    with open("results_jpeg.txt", "w", encoding='utf-8') as file:
        pass
    files = get_file_names("input_image")
    for name in files:
        with open("results_jpeg.txt", "a", encoding='utf-8') as file:
            print('=' * 70, file=file)
            print(' ' * 5, f'Зображення {files.index(name) + 1}: {name}.bmp', file=file)
        len_vyhsdnogo(name, action="write")
        for i in range(1, 3):
            with open("results_jpeg.txt", "a", encoding='utf-8') as file:
                print(f'Таблиця квантування №{i}:\n', file=file)
            encode(name, "results_jpeg", i)
            decoder(name, "results_jpeg", i)
