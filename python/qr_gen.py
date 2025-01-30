# QR code generator from scratch

import numpy as np
import matplotlib.pyplot as plt

class QRCode:

    MIN_VERSION = 1
    MAX_VERSION = 40

    ecc_l = [19, 34, 55, 80, 108, 136, 156, 194, 232, 274, 324, 370, 428, 461, 523, 589, 647, 721, 795, 861, 932, 1006, 1094, 1174, 1276, 1370, 1468, 1531, 1631, 1735, 1843, 1955, 2071, 2191, 2306, 2434, 2566, 2702, 2812]

    ecc_m = [16, 28, 44, 64, 86, 108, 124, 154, 182, 216, 254, 290, 334, 365, 415, 453, 507, 563, 627, 669, 714, 782, 860, 914, 1000, 1062, 1128, 1193, 1267, 1373, 1455, 1541, 1631, 1725, 1812, 1914, 1992, 2102, 2216, 2334]

    ecc_q = [13, 22, 34, 48, 62, 76, 88, 110, 132, 154, 180, 206, 244, 261, 295, 325, 367, 397, 445, 485, 512, 568, 614, 664, 718, 754, 808, 871, 911, 985, 1033, 1115, 1171, 1231, 1286, 1354, 1426, 1502, 1582, 1666]

    ecc_h = [9, 16, 26, 36, 46, 60, 66, 86, 100, 122, 140, 158, 180, 197, 223, 253, 283, 313, 341, 385, 406, 442, 464, 514, 538, 596, 628, 661, 701, 745, 793, 845, 901, 961, 986, 1054, 1096, 1142, 1222, 1276]

    # Static alignment pattern positions for each version
    ALIGNMENT_POSITIONS = {
        1: [],
        2: [6, 18],
        3: [6, 22],
        4: [6, 26],
        5: [6, 30],
        6: [6, 34],
        7: [6, 22, 38],
        8: [6, 24, 42],
        9: [6, 26, 46],
        10: [6, 28, 50]
    }


    def __init__(self, data, version=1, error_correction='L', mask=0) -> None:
        self.data = data
        self.version = version
        self.size = 17 + 4 * self.version
        self.error_correction = error_correction
        self.mask = mask    

        self.modules = np.zeros((self.size, self.size), dtype=int)
        self.isfunction = np.zeros((self.size, self.size), dtype=bool)


        self._add_timing_patterns()
        self._add_finder_patterns()

        if not self.MIN_VERSION <= self.version <= self.MAX_VERSION:
            raise ValueError(f'Version must be between {self.MIN_VERSION} and {self.MAX_VERSION}')
        
        if self.error_correction not in ['L', 'M', 'Q', 'H']:
            raise ValueError('Error correction must be L, M, Q or H')
        
        if not -1 <= self.mask <= 7:
            raise ValueError('Mask must be between 0 and 7')
        
        if mask == -1:
            # TODO: Implement best mask
            for i in range(8):
                pass


    def _add_timing_patterns(self):
        """Add the timing pattern - alternating dark/light modules"""
        for i in range(8, self.size - 8):
            self.modules[6][i] = i % 2  # Horizontal
            self.modules[i][6] = i % 2  # Vertical
            self.isfunction[6][i] = True
            self.isfunction[i][6] = True
            
    def _add_finder_patterns(self):
        """Add the three finder patterns with separators"""
        # Finder pattern - 7x7 modules
        pattern = [
            [1, 1, 1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 0, 1, 1, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 1],
            [1, 0, 1, 1, 1, 0, 1],
            [1, 0, 0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1, 1, 1]
        ]
        
        # Positions for the three finder patterns
        positions = [(0,0), (0, self.size-7), (self.size-7, 0)]
        
        for row, col in positions:
            # Add finder pattern
            for i in range(7):
                for j in range(7):
                    self.modules[row+i][col+j] = pattern[i][j]
                    self.isfunction[row+i][col+j] = True
                    
            # Add white separator border
            for i in range(-1, 8):
                for j in range(-1, 8):
                    if (0 <= row+i < self.size and 0 <= col+j < self.size and
                        (i == -1 or i == 7 or j == -1 or j == 7)):
                        self.modules[row+i][col+j] = 0
                        self.isfunction[row+i][col+j] = True
    
    def optimal_text_segmentation(self):
        """
        Segments the input data optimally into QR Code modes: Numeric, Alphanumeric, Byte, and Kanji.
        """
        s = self.data
        n = len(s)
        v = self.version

        # Mode constants
        MODES = ['numeric', 'alphanumeric', 'byte', 'kanji']
        NUMERIC_SET = set('0123456789')
        ALPHANUMERIC_SET = set('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:')

        # Character count bits based on version
        CHAR_COUNT_BITS = {
            'numeric': [10, 12, 14],
            'alphanumeric': [9, 11, 13],
            'byte': [8, 16, 16],
            'kanji': [8, 10, 12]
        }

        def get_char_count_bits(mode, version):
            if 1 <= version <= 9:
                return CHAR_COUNT_BITS[mode][0]
            elif 10 <= version <= 26:
                return CHAR_COUNT_BITS[mode][1]
            else:
                return CHAR_COUNT_BITS[mode][2]

        def can_encode(c, mode):
            if mode == 'numeric':
                return c in NUMERIC_SET
            elif mode == 'alphanumeric':
                return c in ALPHANUMERIC_SET
            elif mode == 'byte':
                return 0x00 <= ord(c) <= 0xFF
            elif mode == 'kanji':
                return 0x8140 <= ord(c) <= 0x9FFC or 0xE040 <= ord(c) <= 0xEBBF
            return False

        def bit_cost(mode, char_count):
            if mode == 'numeric':
                return (20 * char_count + 5) // 6  # 10/3 bits per char rounded up
            elif mode == 'alphanumeric':
                return (33 * char_count + 5) // 6  # 5.5 bits per char rounded up
            elif mode == 'byte':
                return 8 * char_count  # 8 bits per char
            elif mode == 'kanji':
                return 13 * char_count  # 13 bits per char
            return float('inf')

        # Initialize DP tables
        f = {m: [float('inf')] * (n + 1) for m in MODES}
        g = {m: [float('inf')] * n for m in MODES}
        h = {m: [-1] * (n + 1) for m in MODES}

        # Base case
        for m in MODES:
            f[m][0] = 4 + get_char_count_bits(m, v)

        # Fill DP tables
        for i in range(n):
            for m in MODES:
                if can_encode(s[i], m):
                    g[m][i] = f[m][i] + bit_cost(m, 1)
    
            for m in MODES:
                for prev_m in MODES:
                    cost = (g[prev_m][i] + 7) // 8 * 8  # Round up fractional bits
                    new_cost = cost + 4 + get_char_count_bits(m, v)
                    if new_cost < f[m][i + 1]:
                        f[m][i + 1] = new_cost
                        h[m][i + 1] = prev_m


        # Backtracking to determine optimal segmentation
        segments = []
        m = min(MODES, key=lambda x: f[x][n])  # Find best mode at the end
        i = n
        while i > 0:
            j = i
            while h[m][j] == m:
                j -= 1
            segments.append((m, s[j:i]))
            i = j
            m = h[m][i]

        return list(reversed(segments))

    # Method to get parameters for QR code
    def get_version(self) -> int:
        return self.version
    
    def get_size(self) -> int:
        return self.size
    
    def get_error_correction(self) :
        return self.error_correction
    
    def get_mask(self) -> int:
        return self.mask

    # TODO : Implement mask selection
    def select_mask(self, mask):
        if not -1 <= self.mask <= 7:
            raise ValueError('Mask must be between 0 and 7')
        if mask == -1:
            # TODO: Implement best mask
            pass

    def add_error_correction(self, data: str) -> str:
        """Add error correction to data segment"""
        # Convert binary string to bytes
        data_bytes = bytes(int(data[i:i+8], 2) for i in range(0, len(data), 8))
        
        # Get ECC word count based on version and level
        ecc_words = {
            'L': [7, 10, 15, 20, 26, 36],
            'M': [10, 16, 26, 36, 48, 64],
            'Q': [13, 22, 36, 52, 72, 96],
            'H': [17, 28, 44, 64, 88, 112]
        }[self.error_correction][self.version - 1]
        
        # Generate and append ECC
        rs = ReedSolomon()
        ecc = rs.generate_ecc(data_bytes, ecc_words)
        
        return data + ''.join(f'{b:08b}' for b in ecc)

    def create_data_segment(self, mode) -> str:    
        # Mode constants
        MODES = ['numeric', 'alphanumeric', 'byte', 'kanji']
        if mode not in MODES:
            raise ValueError('Mode must be numeric, alphanumeric, byte or kanji')
        
        # Mode indicator
        mode_indicator = {
            'numeric': '0001',
            'alphanumeric': '0010',
            'byte': '0100',
            'kanji': '1000'
        }

        # Get capacity based on version and ECC level
        capacity_lookup = {
            'L': self.ecc_l,
            'M': self.ecc_m,
            'Q': self.ecc_q,
            'H': self.ecc_h
        }

        total_capacity = capacity_lookup[self.error_correction][self.version - 1] * 8

        # Check if data fits within capacity
        if ( total_capacity // 8 ) < len(self.data):
            raise ValueError(f'Data length exceeds maximum capacity of {total_capacity // 8} bytes for version {self.version} and error correction level {self.error_correction}')

        # Convert data to binary
        binary_data = ''
        for char in self.data:
            binary_data += f'{ord(char):08b}'


        char_count_bits = 8  # Default for byte mode
        bytes_count = bin(len(self.data))[2:].zfill(char_count_bits)
        

        # Add mode indicator, character count, and data bits
        data_bits = mode_indicator[mode] + bytes_count + binary_data
        
        # Add terminator
        remaining = total_capacity - len(data_bits)
        terminator = '0000'[:min(4, remaining)]
        
        # Add bit padding to byte boundary
        remaining -= len(terminator)
        bits_to_boundary = (8 - ((len(data_bits) + len(terminator)) % 8)) % 8
        bit_padding = '0' * min(bits_to_boundary, remaining)
        
        # Add byte padding until capacity reached
        remaining -= len(bit_padding)
        byte_padding = ''
        # 11101100 and 00010001 are repeated alternately 
        while remaining >= 8:
            byte_padding += '11101100' if len(byte_padding)//8 % 2 == 0 else '00010001'
            remaining -= 8
        
        # Segments are concatenated together, without Reed-Solomon error correction
        segment = data_bits + terminator + bit_padding + byte_padding
        # Add error correction
        segment = self.add_error_correction(segment)
        
        return segment

    def draw(self, quiet_zone: int = 0) -> None:
        """
        Draw QR code using matplotlib
        quiet_zone: size of white border (default 0)
        """
        # Add quiet zone to matrix
        size_with_border = self.size + 2 * quiet_zone
        matrix = np.ones((size_with_border, size_with_border))
        
        # Copy QR modules to center of matrix
        matrix[quiet_zone:quiet_zone+self.size, 
               quiet_zone:quiet_zone+self.size] = self.modules
        
        # Create figure with correct proportions
        plt.figure(figsize=(8, 8))
        
        # Draw QR code
        plt.imshow(matrix, cmap='binary')
        
        # Remove axes and margins
        plt.axis('off')
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        
        # Show plot
        plt.show()

class ReedSolomon:
    def __init__(self):
        # GF(256) primitive polynomial x^8 + x^4 + x^3 + x^2 + 1
        self.exp = [1] * 256  # Exponent table
        self.log = [0] * 256  # Log table
        
        # Initialize exp & log tables
        x = 1
        for i in range(255):
            self.exp[i] = x
            self.log[x] = i
            x = x << 1  # Multiply by x
            if x & 0x100:  # Reduce by primitive polynomial
                x ^= 0x11d
    def multiply(self, x: int, y: int) -> int:
        """Multiply in GF(256)"""
        if x == 0 or y == 0:
            return 0
        return self.exp[(self.log[x] + self.log[y]) % 255]
    def generate_ecc(self, data: bytes, ecc_words: int) -> bytes:
        """Generate error correction codewords"""
        # Initialize generator polynomial
        generator = [1]
        for i in range(ecc_words):
            generator = self._multiply_polynomials(generator, [1, self.exp[i]])
        
        # Calculate ECC
        remainder = list(data) + [0] * ecc_words
        for i in range(len(data)):
            factor = remainder[i]
            if factor != 0:
                for j in range(1, len(generator)):
                    remainder[i + j] ^= self.multiply(generator[j], factor)
        
        return bytes(remainder[-ecc_words:])
    def _multiply_polynomials(self, p1: list, p2: list) -> list:
        """Multiply two polynomials in GF(256)"""
        result = [0] * (len(p1) + len(p2) - 1)
        for i in range(len(p1)):
            for j in range(len(p2)):
                result[i + j] ^= self.multiply(p1[i], p2[j])
        return result

text = 'Hello, world! 123'
qr = QRCode(text, version= 3, error_correction= 'L')

x = qr.create_data_segment('byte')
# # Print hex value of binary string 2 bytes at a time
# for i in range(0, len(x), 8):
#     print(hex(int(x[i:i+8], 2))[2:], end=' ')
# print()
# # print(list(plt.colormaps))
# print(x)
qr.draw()