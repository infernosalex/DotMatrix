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

    # Block structure tables for versions 1-40 (version, error correction): (num_blocks_g1, words_per_block_g1, num_blocks_g2, words_per_block_g2)
    BLOCK_STRUCTURE = {
        # Version 1
        (1, 'L'): (1, 19, 0, 0),
        (1, 'M'): (1, 16, 0, 0),
        (1, 'Q'): (1, 13, 0, 0),
        (1, 'H'): (1, 9, 0, 0),
        # Version 2
        (2, 'L'): (1, 34, 0, 0),
        (2, 'M'): (1, 28, 0, 0),
        (2, 'Q'): (1, 22, 0, 0),
        (2, 'H'): (1, 16, 0, 0),
        # Version 3
        (3, 'L'): (1, 55, 0, 0),
        (3, 'M'): (1, 44, 0, 0),
        (3, 'Q'): (2, 17, 0, 0),
        (3, 'H'): (2, 13, 0, 0),
        # Version 4
        (4, 'L'): (1, 80, 0, 0),
        (4, 'M'): (2, 32, 0, 0),
        (4, 'Q'): (2, 24, 0, 0),
        (4, 'H'): (4, 9, 0, 0),
        # Version 5
        (5, 'L'): (1, 108, 0, 0),
        (5, 'M'): (2, 43, 0, 0),
        (5, 'Q'): (2, 15, 2, 16),
        (5, 'H'): (2, 11, 2, 12),
        # Version 6
        (6, 'L'): (2, 68, 0, 0),
        (6, 'M'): (4, 27, 0, 0),
        (6, 'Q'): (4, 19, 0, 0),
        (6, 'H'): (4, 15, 0, 0),
        # Version 7
        (7, 'L'): (2, 78, 0, 0),
        (7, 'M'): (4, 31, 0, 0),
        (7, 'Q'): (2, 14, 4, 15),
        (7, 'H'): (4, 13, 1, 14),
        # Version 8
        (8, 'L'): (2, 97, 0, 0),
        (8, 'M'): (2, 38, 2, 39),
        (8, 'Q'): (4, 18, 2, 19),
        (8, 'H'): (4, 14, 2, 15),
        # Version 9
        (9, 'L'): (2, 116, 0, 0),
        (9, 'M'): (3, 36, 2, 37),
        (9, 'Q'): (4, 16, 4, 17),
        (9, 'H'): (4, 12, 4, 13),
        # Version 10
        (10, 'L'): (2, 68, 2, 69),
        (10, 'M'): (4, 43, 1, 44),
        (10, 'Q'): (6, 19, 2, 20),
        (10, 'H'): (6, 15, 2, 16),
        # Version 11
        (11, 'L'): (4, 81, 0, 0),
        (11, 'M'): (1, 50, 4, 51),
        (11, 'Q'): (4, 22, 4, 23),
        (11, 'H'): (3, 12, 8, 13),
        # Version 12
        (12, 'L'): (2, 92, 2, 93),
        (12, 'M'): (6, 36, 2, 37),
        (12, 'Q'): (4, 20, 6, 21),
        (12, 'H'): (7, 14, 4, 15),
        # Version 13
        (13, 'L'): (4, 107, 0, 0),
        (13, 'M'): (8, 37, 1, 38),
        (13, 'Q'): (8, 20, 4, 21),
        (13, 'H'): (12, 11, 4, 12),
        # Version 14
        (14, 'L'): (3, 115, 1, 116),
        (14, 'M'): (4, 40, 5, 41),
        (14, 'Q'): (11, 16, 5, 17),
        (14, 'H'): (11, 12, 5, 13),
        # Version 15
        (15, 'L'): (5, 87, 1, 88),
        (15, 'M'): (5, 41, 5, 42),
        (15, 'Q'): (5, 24, 7, 25),
        (15, 'H'): (11, 12, 7, 13),
        # Version 16
        (16, 'L'): (5, 98, 1, 99),
        (16, 'M'): (7, 45, 3, 46),
        (16, 'Q'): (15, 19, 2, 20),
        (16, 'H'): (3, 15, 13, 16),
        # Version 17
        (17, 'L'): (1, 107, 5, 108),
        (17, 'M'): (10, 46, 1, 47),
        (17, 'Q'): (1, 22, 15, 23),
        (17, 'H'): (2, 14, 17, 15),
        # Version 18
        (18, 'L'): (5, 120, 1, 121),
        (18, 'M'): (9, 43, 4, 44),
        (18, 'Q'): (17, 22, 1, 23),
        (18, 'H'): (2, 14, 19, 15),
        # Version 19
        (19, 'L'): (3, 113, 4, 114),
        (19, 'M'): (3, 44, 11, 45),
        (19, 'Q'): (17, 21, 4, 22),
        (19, 'H'): (9, 13, 16, 14),
        # Version 20
        (20, 'L'): (3, 107, 5, 108),
        (20, 'M'): (3, 41, 13, 42),
        (20, 'Q'): (15, 24, 5, 25),
        (20, 'H'): (15, 15, 10, 16),
        # Version 21
        (21, 'L'): (4, 116, 4, 117),
        (21, 'M'): (17, 42, 0, 0),
        (21, 'Q'): (17, 22, 6, 23),
        (21, 'H'): (19, 16, 6, 17),
        # Version 22
        (22, 'L'): (2, 111, 7, 112),
        (22, 'M'): (17, 46, 0, 0),
        (22, 'Q'): (7, 24, 16, 25),
        (22, 'H'): (34, 13, 0, 0),
        # Version 23
        (23, 'L'): (4, 121, 5, 122),
        (23, 'M'): (4, 47, 14, 48),
        (23, 'Q'): (11, 24, 14, 25),
        (23, 'H'): (16, 15, 14, 16),
        # Version 24
        (24, 'L'): (6, 117, 4, 118),
        (24, 'M'): (6, 45, 14, 46),
        (24, 'Q'): (11, 24, 16, 25),
        (24, 'H'): (30, 16, 2, 17),
        # Version 25
        (25, 'L'): (8, 106, 4, 107),
        (25, 'M'): (8, 47, 13, 48),
        (25, 'Q'): (7, 24, 22, 25),
        (25, 'H'): (22, 15, 13, 16),
        # Version 26
        (26, 'L'): (10, 114, 2, 115),
        (26, 'M'): (19, 46, 4, 47),
        (26, 'Q'): (28, 22, 6, 23),
        (26, 'H'): (33, 16, 4, 17),
        # Version 27
        (27, 'L'): (8, 122, 4, 123),
        (27, 'M'): (22, 45, 3, 46),
        (27, 'Q'): (8, 23, 26, 24),
        (27, 'H'): (12, 15, 28, 16),
        # Version 28
        (28, 'L'): (3, 117, 10, 118),
        (28, 'M'): (3, 45, 23, 46),
        (28, 'Q'): (4, 24, 31, 25),
        (28, 'H'): (11, 15, 31, 16),
        # Version 29
        (29, 'L'): (7, 116, 7, 117),
        (29, 'M'): (21, 45, 7, 46),
        (29, 'Q'): (1, 23, 37, 24),
        (29, 'H'): (19, 15, 26, 16),
        # Version 30
        (30, 'L'): (5, 115, 10, 116),
        (30, 'M'): (19, 47, 10, 48),
        (30, 'Q'): (15, 24, 25, 25),
        (30, 'H'): (23, 15, 25, 16),
        # Version 31
        (31, 'L'): (13, 115, 3, 116),
        (31, 'M'): (2, 46, 29, 47),
        (31, 'Q'): (42, 24, 1, 25),
        (31, 'H'): (23, 15, 28, 16),
        # Version 32
        (32, 'L'): (17, 115, 0, 0),
        (32, 'M'): (10, 46, 23, 47),
        (32, 'Q'): (10, 24, 35, 25),
        (32, 'H'): (19, 15, 35, 16),
        # Version 33
        (33, 'L'): (17, 115, 1, 116),
        (33, 'M'): (14, 46, 21, 47),
        (33, 'Q'): (29, 24, 19, 25),
        (33, 'H'): (11, 15, 46, 16),
        # Version 34
        (34, 'L'): (13, 115, 6, 116),
        (34, 'M'): (14, 46, 23, 47),
        (34, 'Q'): (44, 24, 7, 25),
        (34, 'H'): (59, 16, 1, 17),
        # Version 35
        (35, 'L'): (12, 121, 7, 122),
        (35, 'M'): (12, 47, 26, 48),
        (35, 'Q'): (39, 24, 14, 25),
        (35, 'H'): (22, 15, 41, 16),
        # Version 36
        (36, 'L'): (6, 121, 14, 122),
        (36, 'M'): (6, 47, 34, 48),
        (36, 'Q'): (46, 24, 10, 25),
        (36, 'H'): (2, 15, 64, 16),
        # Version 37
        (37, 'L'): (17, 122, 4, 123),
        (37, 'M'): (29, 46, 14, 47),
        (37, 'Q'): (49, 24, 10, 25),
        (37, 'H'): (24, 15, 46, 16),
        # Version 38
        (38, 'L'): (4, 122, 18, 123),
        (38, 'M'): (13, 46, 32, 47),
        (38, 'Q'): (48, 24, 14, 25),
        (38, 'H'): (42, 15, 32, 16),
        # Version 39
        (39, 'L'): (20, 117, 4, 118),
        (39, 'M'): (40, 47, 7, 48),
        (39, 'Q'): (43, 24, 22, 25),
        (39, 'H'): (10, 15, 67, 16),
        # Version 40
        (40, 'L'): (19, 118, 6, 119),
        (40, 'M'): (18, 47, 31, 48),
        (40, 'Q'): (34, 24, 34, 25),
        (40, 'H'): (20, 15, 61, 16)
    }

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
        10: [6, 28, 50],
        11: [6, 30, 54],
        12: [6, 32, 58],
        13: [6, 34, 62],
        14: [6, 26, 46, 66],
        15: [6, 26, 48, 70],
        16: [6, 26, 50, 74],
        17: [6, 30, 54, 78],
        18: [6, 30, 56, 82],
        19: [6, 30, 58, 86],
        20: [6, 34, 62, 90],
        21: [6, 28, 50, 72, 94],
        22: [6, 26, 50, 74, 98],
        23: [6, 30, 54, 78, 102],
        24: [6, 28, 54, 80, 106],
        25: [6, 32, 58, 84, 110],
        26: [6, 30, 58, 86, 114],
        27: [6, 34, 62, 90, 118],
        28: [6, 26, 50, 74, 98, 122],
        29: [6, 30, 54, 78, 102, 126],
        30: [6, 26, 52, 78, 104, 130],
        31: [6, 30, 56, 82, 108, 134],
        32: [6, 34, 60, 86, 112, 138],
        33: [6, 30, 58, 86, 114, 142],
        34: [6, 34, 62, 90, 118, 146],
        35: [6, 30, 54, 78, 102, 126, 150],
        36: [6, 24, 50, 76, 102, 128, 154],
        37: [6, 28, 54, 80, 106, 132, 158],
        38: [6, 32, 58, 84, 110, 136, 162],
        39: [6, 26, 54, 82, 110, 138, 166],
        40: [6, 30, 58, 86, 114, 142, 170]
    }

    # Version information bit strings for versions 7-40
    VERSION_INFO = {
        7: 0b000111110010010100,
        8: 0b001000010110111100,
        9: 0b001001101010011001,
        10: 0b001010010011010011,
        11: 0b001011101111110110,
        12: 0b001100011101100010,
        13: 0b001101100001000111,
        14: 0b001110011000001101,
        15: 0b001111100100101000,
        16: 0b010000101101111000,
        17: 0b010001010001011101,
        18: 0b010010101000010111,
        19: 0b010011010100110010,
        20: 0b010100100110100110,
        21: 0b010101011010000011,
        22: 0b010110100011001001,
        23: 0b010111011111101100,
        24: 0b011000111011000100,
        25: 0b011001000111100001,
        26: 0b011010111110101011,
        27: 0b011011000010001110,
        28: 0b011100110000011010,
        29: 0b011101001100111111,
        30: 0b011110110101110101,
        31: 0b011111001001010000,
        32: 0b100000100111010101,
        33: 0b100001011011110000,
        34: 0b100010100010111010,
        35: 0b100011011110011111,
        36: 0b100100101100001011,
        37: 0b100101010000101110,
        38: 0b100110101001100100,
        39: 0b100111010101000001,
        40: 0b101000110001101001
    }

    def __init__(self, data, version=-1, error_correction='L', mask=0, debug=False, mode='auto') -> None:
        self.data = data
        self.error_correction = error_correction
        self.mask = mask
        self.debug = debug
        self.mode = mode

        # Validate error correction level
        if self.error_correction not in ['L', 'M', 'Q', 'H']:
            raise ValueError('Error correction must be L, M, Q or H')
        
        if not -1 <= self.mask <= 7:
            raise ValueError('Mask must be between 0 and 7')

        # Auto-select version if -1
        if version == -1:
            self.version = self._select_version()
        else:
            self.version = version
            if not self.MIN_VERSION <= self.version <= self.MAX_VERSION:
                raise ValueError(f'Version must be between {self.MIN_VERSION} and {self.MAX_VERSION}')

        self.size = 17 + 4 * self.version
        self.modules = np.zeros((self.size, self.size), dtype=int)
        self.isfunction = np.zeros((self.size, self.size), dtype=bool)

        # Add all function patterns first
        self._add_temporary_format_bits()
        self._add_timing_patterns()
        self._add_finder_patterns()
        self._add_alignment_patterns()
        if self.version >= 7:
            self._add_version_information()

        # Create and place data
        data_bits = self.create_data_segment(self.mode)
        data_bits = self.add_error_correction(data_bits)  # Add error correction after segmentation
        self._place_data_bits(data_bits)
        
        if mask == -1:
            self.select_mask()
        else:
            self._apply_mask(mask)
            self._draw_format_bits(self.mask)  # Draw final format bits

    def _add_timing_patterns(self):
        """Add the timing pattern - alternating dark/light modules"""
        # horizontal timing pattern
        for i in range(0, self.size):
            self.modules[6, i] = (i+1) % 2
            self.isfunction[6, i] = True

        # vertical timing pattern
        for i in range(0, self.size):
            self.modules[i, 6] = (i+1) % 2
            self.isfunction[i, 6] = True

    def _add_finder_patterns(self):
        """Add the three finder patterns with separators"""
        # Helper function to add a finder pattern at given position
        def add_finder_pattern(row, col):
            # Add the 7x7 finder pattern
            for r in range(-3, 4):
                for c in range(-3, 4):
                    # Create the pattern: 3x3 black square inside 5x5 white square inside 7x7 black border
                    is_border = abs(r) == 3 or abs(c) == 3
                    is_inner = abs(r) <= 1 and abs(c) <= 1
                    self.modules[row + r, col + c] = 1 if (is_border or is_inner) else 0
                    self.isfunction[row + r, col + c] = True
            
            # Add separator (white border)
            for r in range(-4, 5):
                for c in range(-4, 5):
                    if abs(r) == 4 or abs(c) == 4:
                        if 0 <= row + r < self.size and 0 <= col + c < self.size:
                            self.modules[row + r, col + c] = 0
                            self.isfunction[row + r, col + c] = True

        # Add finder patterns at the three corners
        add_finder_pattern(3, 3)  # Top-left
        add_finder_pattern(3, self.size - 4)  # Top-right
        add_finder_pattern(self.size - 4, 3)  # Bottom-left

    def _add_alignment_patterns(self):
        """Add alignment patterns in standard positions, skipping finder corners"""
        # Get alignment pattern positions for current version
        if self.version < 2:  # Version 1 has no alignment patterns
            return
            
        positions = self.ALIGNMENT_POSITIONS[self.version]
        
        # Add alignment patterns at all intersections of positions
        for row in positions:
            for col in positions:
                # Skip if too close to finder patterns
                if ((row <= 8 and col <= 8) or      # Top-left
                    (row <= 8 and col >= self.size - 9) or    # Top-right
                    (row >= self.size - 9 and col <= 8)):      # Bottom-left
                    continue
                    
                # Add 5x5 alignment pattern
                for r in range(-2, 3):
                    for c in range(-2, 3):
                        # Create pattern: single black center, white square, black border
                        is_border = abs(r) == 2 or abs(c) == 2
                        is_center = r == 0 and c == 0
                        self.modules[row + r, col + c] = 1 if (is_border or is_center) else 0
                        self.isfunction[row + r, col + c] = True

    def _add_temporary_format_bits(self):
        """Add dummy format information bits (will be replaced later)"""
        # Around top-left finder pattern
        for i in range(9):
            for j in range(9):
                self.modules[i, j] = 0
                self.isfunction[i, j] = True

        # Below/beside top-right finder pattern
        for i in range(9):
            for j in range(8):
                self.modules[i, self.size - 1 - j] = 0
                self.isfunction[i, self.size - 1 - j] = True

        # Below/beside bottom-left finder pattern
        for i in range(8):
            for j in range(9):
                self.modules[self.size - 1 - i, j] = 0
                self.isfunction[self.size - 1 - i, j] = True

        # Dark module
        self.modules[self.size - 8, 8] = 1
        self.isfunction[self.size - 8, 8] = True

    def _place_data_bits(self, data_bits):
        """Place data bits in zigzag pattern from bottom right"""
        # Convert binary string to bytes for easier bit access
        data_bytes = bytes(int(data_bits[i:i+8], 2) for i in range(0, len(data_bits), 8))
        
        bit_idx = 0  # Bit index into the data
        total_bits = len(data_bits)

        # Start from the bottom right corner
        # Move upward in pairs of columns from right to left
        for right in range(self.size - 1, 0, -2):  # Index of right column in each column pair
            # Timing pattern adjustment
            if right <= 6:
                right -= 1
                
            # For each column pair, traverse vertically
            for vert in range(self.size):
                # Process both columns in the pair
                for j in range(2):
                    x = right - j  # Actual x coordinate
                    
                    # Determine if we're moving upward or downward in this column
                    # Changes direction every two columns (when right + 1 is divisible by 4)
                    upward = ((right + 1) & 2) == 0
                    y = (self.size - 1 - vert) if upward else vert  # Actual y coordinate
                    
                    # Skip if this is a function module or we've placed all bits
                    if not self.isfunction[y, x] and bit_idx < total_bits:
                        # Get the current bit from the data
                        # First get the byte index (bit_idx // 8)
                        # Then get the bit position within that byte (7 - (bit_idx % 8))
                        current_bit = (data_bytes[bit_idx // 8] >> (7 - (bit_idx % 8))) & 1
                        self.modules[y, x] = current_bit
                        bit_idx += 1
                        
        # Any remaining modules should stay as initialized (white/0)
    # Method to get parameters for QR code
    def get_version(self) -> int:
        return self.version
    
    def get_size(self) -> int:
        return self.size
    
    def get_error_correction(self) -> str:
        return self.error_correction
    
    def get_mask(self) -> int:
        return self.mask

    def select_mask(self) -> int:
        """Select the best mask pattern based on penalty scores."""
        min_score = float('inf')
        best_mask = 0
        original_modules = self.modules.copy()  # Save original state

        for mask in range(8):
            self.modules = original_modules.copy()  # Restore original state
            self._apply_mask(mask)
            self._draw_format_bits(mask)  # Draw format bits for this mask
            score = self._get_penalty_score()

            if self.debug:
                print(f'Mask {mask} score: {score}')
            
            if score < min_score:
                min_score = score
                best_mask = mask
            
            # Undo the mask for next iteration
            self._apply_mask(mask)

        # Apply the best mask
        self.modules = original_modules
        self._apply_mask(best_mask)
        self._draw_format_bits(best_mask)  # Draw final format bits
        self.mask = best_mask
        return best_mask

    def add_error_correction(self, data: str) -> str:
        """Add error correction to data segment"""
        # Convert binary string to bytes
        data_bytes = bytes(int(data[i:i+8], 2) for i in range(0, len(data), 8))
        
        # Get block structure
        block_structure = self.BLOCK_STRUCTURE.get((self.version, self.error_correction))
        if not block_structure:
            raise ValueError(f"Block structure not defined for version {self.version} and error correction level {self.error_correction}")
        
        num_blocks_g1, words_per_block_g1, num_blocks_g2, words_per_block_g2 = block_structure
        
        # Print debug information
        if self.debug:
            print("\nBlock Structure:")
            print(f"Number of blocks in group 1: {num_blocks_g1}")
            print(f"Words per block in group 1: {words_per_block_g1}")
            print(f"Number of blocks in group 2: {num_blocks_g2}")
            print(f"Words per block in group 2: {words_per_block_g2}")
        
        # Get ECC word count based on version and level
        ecc_words = {
            'L': [7,10,15,20,26,18,20,24,30,18,20,24,26,30,22,24,28,30,28,28,28,28,30,30,26,28,30,30,30,30,30,30,30,30,30,30,30,30,30],
            'M': [10,16,26,18,24,16,18,22,22,26,30,22,22,24,24,28,28,26,26,26,26,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28,28],
            'Q': [13,22,18,26,18,24,18,22,20,24,28,26,24,20,30,24,28,28,26,30,28,30,30,30,30,28,30,30,30,30,30,30,30,30,30,30,30,30,30], 
            'H': [17,28,22,16,22,28,26,26,24,28,24,28,22,24,24,30,28,28,26,28,30,24,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30,30]
        }[self.error_correction][self.version - 1]
        if self.debug:
            print(f"ECC words per block: {ecc_words}")

        # Split data into blocks
        blocks = []
        pos = 0
        
        # Group 1 blocks
        for _ in range(num_blocks_g1):
            block = list(data_bytes[pos:pos + words_per_block_g1])
            blocks.append(block)
            pos += words_per_block_g1
            
        # Group 2 blocks (if any)
        for _ in range(num_blocks_g2):
            block = list(data_bytes[pos:pos + words_per_block_g2])
            blocks.append(block)
            pos += words_per_block_g2

        # Generate ECC for each block
        rs = ReedSolomon()
        ecc_blocks = []
        for i, block in enumerate(blocks):
            ecc = list(rs.generate_ecc(bytes(block), ecc_words))
            ecc_blocks.append(ecc)
            if self.debug:
                print(f"\nBlock {i} data:", ' '.join(f'{b:02X}' for b in block))
                print(f"Block {i} ECC:", ' '.join(f'{b:02X}' for b in ecc))

        # Interleave data codewords
        final_data = bytearray()
        max_data_words = max(words_per_block_g1, words_per_block_g2 if words_per_block_g2 else 0)
        
        # Interleave data codewords
        for i in range(max_data_words):
            for block in blocks:
                if i < len(block):
                    final_data.append(block[i])

        # Interleave error correction codewords
        for i in range(ecc_words):
            for ecc_block in ecc_blocks:
                final_data.append(ecc_block[i])

        if self.debug:
            print("\nFinal interleaved sequence:", ' '.join(f'{b:02X}' for b in final_data))
        
        # Convert the entire final sequence to binary string
        final_bits = ''.join(f'{b:08b}' for b in final_data)
        return final_bits

    def create_data_segment(self, mode) -> str:
        """Create the data segment with mode indicator and character count"""
        # Mode constants
        MODES = ['numeric', 'alphanumeric', 'byte', 'kanji', 'auto']
        if mode not in MODES:
            raise ValueError('Mode must be numeric, alphanumeric, byte, kanji or auto')
        
        if self.debug:
            print("\n=== QR Code Generation Debug ===")

        # Get capacity based on version and ECC level
        capacity_lookup = {
            'L': self.ecc_l,
            'M': self.ecc_m,
            'Q': self.ecc_q,
            'H': self.ecc_h
        }

        total_capacity = capacity_lookup[self.error_correction][self.version - 1] * 8
        if self.debug:
            print(f"Total capacity: {total_capacity} bits")

        # If auto mode is selected, use the segmentation algorithm
        if mode == 'auto':
            from tokens import optimize_segments, Mode as QRMode
            segments = optimize_segments(self.data, self.version)

            # Convert segments to bit string
            data_bits = ""
            
            for segment in segments:
                # Add mode indicator (4 bits)
                if segment.mode == QRMode.NUMERIC:
                    data_bits += "0001"  # Numeric mode
                    count_bits = format(len(segment.data), f'0{self.get_mode_bits(QRMode.NUMERIC, self.version)}b')
                    data_bits += count_bits
                    # Process 3 digits at a time
                    data = segment.data
                    for i in range(0, len(data), 3):
                        chunk = data[i:i+3]
                        value = int(chunk)
                        if len(chunk) == 3:
                            data_bits += format(value, '010b')  # 10 bits for 3 digits
                        elif len(chunk) == 2:
                            data_bits += format(value, '07b')   # 7 bits for 2 digits
                        else:
                            data_bits += format(value, '04b')   # 4 bits for 1 digit
                
                elif segment.mode == QRMode.ALPHANUMERIC:
                    data_bits += "0010"  # Alphanumeric mode
                    count_bits = format(len(segment.data), f'0{self.get_mode_bits(QRMode.ALPHANUMERIC, self.version)}b')
                    data_bits += count_bits
                    # Process 2 characters at a time
                    data = segment.data
                    for i in range(0, len(data), 2):
                        if i + 1 < len(data):
                            char1 = self._alphanumeric_value(data[i])
                            char2 = self._alphanumeric_value(data[i + 1])
                            value = char1 * 45 + char2
                            data_bits += format(value, '011b')  # 11 bits for 2 chars
                        else:
                            char1 = self._alphanumeric_value(data[i])
                            data_bits += format(char1, '06b')   # 6 bits for 1 char
                
                elif segment.mode == QRMode.BYTE:
                    data_bits += "0100"  # Byte mode
                    count_bits = format(len(segment.data), f'0{self.get_mode_bits(QRMode.BYTE, self.version)}b')
                    data_bits += count_bits
                    for char in segment.data:
                        data_bits += format(ord(char), '08b')
                
                elif segment.mode == QRMode.KANJI:
                    data_bits += "1000"  # Kanji mode
                    count_bits = format(len(segment.data), f'0{self.get_mode_bits(QRMode.KANJI, self.version)}b')
                    data_bits += count_bits
                    for char in segment.data:
                        value = ord(char)
                        data_bits += format(value, '013b')

            if self.debug:
                print(f"Data bits from segmentation: {data_bits}")
                print(f"Data length: {len(data_bits)} bits")
        else:
            # Use original single-mode encoding
            mode_indicator = {
                'numeric': '0001',
                'alphanumeric': '0010',
                'byte': '0100',
                'kanji': '1000'
            }
            if self.debug:
                print(f"Mode: {mode} (indicator: {mode_indicator[mode]})")

            # Check if data fits within capacity
            if (total_capacity // 8) < len(self.data):
                raise ValueError(f'Data length exceeds maximum capacity of {total_capacity // 8} bytes for version {self.version} and error correction level {self.error_correction}')

            # Get character count bits based on version and mode
            from tokens import Mode as QRMode
            mode_enum = {
                'numeric': QRMode.NUMERIC,
                'alphanumeric': QRMode.ALPHANUMERIC,
                'byte': QRMode.BYTE,
                'kanji': QRMode.KANJI
            }[mode]
            char_count_bits = self.get_mode_bits(mode_enum, self.version)

            # Convert data to binary
            binary_data = ''
            for char in self.data:
                binary_data += f'{ord(char):08b}'
            if self.debug:
                print(f"Data length: {len(self.data)} bytes")
                print(f"Binary data: {binary_data}")

            bytes_count = format(len(self.data), f'0{char_count_bits}b')
            if self.debug:
                print(f"Character count indicator ({char_count_bits} bits): {bytes_count}")

            # Add mode indicator, character count, and data bits
            data_bits = mode_indicator[mode] + bytes_count + binary_data
            if self.debug:
                print(f"\nInitial data bits: {data_bits}")
                print(f"Initial data length: {len(data_bits)} bits")

        # Add terminator
        remaining = total_capacity - len(data_bits)
        terminator = '0000'[:min(4, remaining)]
        if self.debug:
            print(f"\nAdding terminator: {terminator}")
        
        # Add bit padding to byte boundary
        remaining -= len(terminator)
        bits_to_boundary = (8 - ((len(data_bits) + len(terminator)) % 8)) % 8
        bit_padding = '0' * min(bits_to_boundary, remaining)
        if self.debug:
            print(f"Adding bit padding: {bit_padding}")
        
        # Add byte padding until capacity reached
        remaining -= len(bit_padding)
        byte_padding = ''
        # 11101100 and 00010001 are repeated alternately 
        while remaining >= 8:
            byte_padding += '11101100' if len(byte_padding)//8 % 2 == 0 else '00010001'
            remaining -= 8
        if self.debug:
            print(f"Adding byte padding: {byte_padding}")
        
        # Segments are concatenated together, without Reed-Solomon error correction
        segment = data_bits + terminator + bit_padding + byte_padding
        if self.debug:
            print(f"\nFinal data segment (before error correction): {segment}")
            print(f"Final segment length: {len(segment)} bits")
        
        return segment

    def _alphanumeric_value(self, char: str) -> int:
        """Convert alphanumeric character to its value"""
        if char.isdigit():
            return int(char)
        elif char.isupper():
            return ord(char) - ord('A') + 10
        elif char == ' ':
            return 36
        elif char == '$':
            return 37
        elif char == '%':
            return 38
        elif char == '*':
            return 39
        elif char == '+':
            return 40
        elif char == '-':
            return 41
        elif char == '.':
            return 42
        elif char == '/':
            return 43
        elif char == ':':
            return 44
        else:
            raise ValueError(f"Invalid alphanumeric character: {char}")

    def _get_data_codewords_count(self) -> int:
        """Get the number of data codewords for current version and error correction level"""
        if self.error_correction == 'L':
            return self.ecc_l[self.version - 1]
        elif self.error_correction == 'M':
            return self.ecc_m[self.version - 1]
        elif self.error_correction == 'Q':
            return self.ecc_q[self.version - 1]
        else:  # H
            return self.ecc_h[self.version - 1]

    def draw(self, quiet_zone: int = 0) -> None:
        """
        Draw QR code using matplotlib
        quiet_zone: size of white border (default 0)
        """
        # Add quiet zone if requested
        if quiet_zone > 0:
            matrix = np.pad(self.modules, quiet_zone, mode='constant', constant_values=0)
            is_function = np.pad(self.isfunction, quiet_zone, mode='constant', constant_values=False)
        else:
            matrix = self.modules
            is_function = self.isfunction

        # Create figure with correct proportions
        plt.figure(figsize=(8, 8))
        
        # Create a masked array where non-function bits are masked
        masked_matrix = np.ma.masked_array(matrix, mask=~is_function)
        
        # Draw the background orange first
        plt.imshow(np.ones_like(matrix), cmap='Oranges', vmin=0, vmax=1)
        
        # Draw QR code functional patterns in black and white
        plt.imshow(masked_matrix, cmap='binary')
        
        # Remove axes and margins
        plt.axis('off')
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        
        # Show plot
        plt.show()

    def _apply_mask(self, mask: int) -> None:
        """Apply the specified mask pattern to the QR code."""
        if not 0 <= mask <= 7:
            raise ValueError("Mask value out of range")
        
        # Mask patterns (x and y are module coordinates)
        patterns = [
            lambda x, y: (x + y) % 2 == 0,                    # 000
            lambda x, y: y % 2 == 0,                          # 001
            lambda x, y: x % 3 == 0,                          # 010
            lambda x, y: (x + y) % 3 == 0,                    # 011
            lambda x, y: ((y // 2) + (x // 3)) % 2 == 0,      # 100
            lambda x, y: ((x * y) % 2) + ((x * y) % 3) == 0,  # 101
            lambda x, y: (((x * y) % 2) + ((x * y) % 3)) % 2 == 0,  # 110
            lambda x, y: (((x + y) % 2) + ((x * y) % 3)) % 2 == 0,  # 111
        ]
        
        # Apply mask to all non-function modules
        for y in range(self.size):
            for x in range(self.size):
                if not self.isfunction[y][x]:
                    self.modules[y][x] ^= patterns[mask](x, y)

    def _get_penalty_score(self) -> int:
        """Calculate penalty score for a mask pattern based on QR code specification."""
        score = 0
        size = self.size
        modules = self.modules

        # Rule 1: Adjacent modules in row/column in same color (runs)
        def count_runs(row):
            run_length = 1
            run_score = 0
            prev_color = row[0]
            
            for i in range(1, len(row)):
                if row[i] == prev_color:
                    run_length += 1
                else:
                    if run_length >= 5:
                        run_score += 3 + (run_length - 5)  # 3 points for 5 modules, +1 for each additional
                    run_length = 1
                    prev_color = row[i]
            
            # Check final run
            if run_length >= 5:
                run_score += 3 + (run_length - 5)
            
            return run_score

        # Check horizontal runs
        for y in range(size):
            score += count_runs(modules[y])

        # Check vertical runs
        for x in range(size):
            score += count_runs(modules[:, x])

        # Rule 2: 2×2 blocks of same color
        box_score = 0
        for y in range(size - 1):
            for x in range(size - 1):
                if modules[y, x] == modules[y, x+1] == modules[y+1, x] == modules[y+1, x+1]:
                    box_score += 3

        score += box_score

        # Rule 3: Finder-like patterns
        finder_pattern = np.array([1, 0, 1, 1, 1, 0, 1])
        finder_score = 0

        # Check horizontal finder patterns
        for y in range(size):
            for x in range(size - 6):
                if np.array_equal(modules[y, x:x+7], finder_pattern):
                    finder_score += 40
                # Check reverse pattern too
                if np.array_equal(modules[y, x:x+7], finder_pattern[::-1]):
                    finder_score += 40

        # Check vertical finder patterns
        for x in range(size):
            for y in range(size - 6):
                if np.array_equal(modules[y:y+7, x], finder_pattern):
                    finder_score += 40
                # Check reverse pattern too
                if np.array_equal(modules[y:y+7, x], finder_pattern[::-1]):
                    finder_score += 40

        score += finder_score

        # Rule 4: Balance of dark/light modules
        dark_count = np.sum(modules)
        total = size * size
        dark_percentage = (dark_count * 100) / total
        
        # Calculate deviation from 50%
        deviation = abs(dark_percentage - 50)
        balance_score = 10 * ((deviation + 4) // 5)  # Integer division by 5, then multiply by 10
        
        score += balance_score

        return score

    def draw(self) -> None:
        """Draw QR code after applying mask"""
        plt.figure(figsize=(8, 8))
        plt.imshow(np.ones_like(self.modules), cmap='Oranges', vmin=0, vmax=1)
        plt.imshow(self.modules, cmap='binary')
        plt.axis('off')
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.title(f'QR Code (Mask Pattern: {self.mask} Version: {self.version} ECC: {self.error_correction})')
        plt.show()

    def _draw_format_bits(self, mask: int) -> None:
        """Draw the format bits (error correction level and mask pattern) with error correction."""
        # Format info is 15 bits: 2 bits for error correction level, 3 bits for mask,
        # 10 bits for error correction of format information itself
        format_bits = {
            'L': 0b01,
            'M': 0b00,
            'Q': 0b11,
            'H': 0b10
        }[self.error_correction] << 3 | mask

        # Calculate error correction for format bits
        format_ecc = format_bits
        for _ in range(10):
            format_ecc = (format_ecc << 1) ^ ((format_ecc >> 9) * 0x537)
        format_data = ((format_bits << 10) | format_ecc) ^ 0x5412  # XOR with mask pattern

        # Draw format bits around the top-left finder pattern
        # Vertical bits
        for i in range(8):
            bit = (format_data >> i) & 1
            # Skip the timing pattern at position 6
            row = i if i < 6 else i + 1
            self.modules[row, 8] = bit
            self.isfunction[row, 8] = True

        # Horizontal bits
        for i in range(8):
            bit = (format_data >> (14 - i)) & 1
            # Skip the timing pattern at position 6
            col = i if i < 6 else i + 1
            self.modules[8, col] = bit
            self.isfunction[8, col] = True

        # Draw format bits around the top-right and bottom-left finder patterns
        # Top right
        for i in range(8):
            bit = (format_data >> i) & 1
            self.modules[8, self.size - 1 - i] = bit
            self.isfunction[8, self.size - 1 - i] = True

        # Bottom left
        for i in range(8):
            bit = (format_data >> (14 - i)) & 1
            self.modules[self.size - i - 1, 8] = bit
            self.isfunction[self.size - i - 1, 8] = True

        # Always set the dark module
        self.modules[self.size - 8, 8] = 1
        self.isfunction[self.size - 8, 8] = True

    def _select_version(self) -> int:
        """Select the smallest version that can hold the data with the given error correction level."""
        # Get capacity array based on error correction level
        capacity_lookup = {
            'L': self.ecc_l,
            'M': self.ecc_m,
            'Q': self.ecc_q,
            'H': self.ecc_h
        }
        capacities = capacity_lookup[self.error_correction]

        # Calculate required capacity in bytes
        # For byte mode: mode indicator (4 bits) + character count indicator (8 bits) + data (8 bits per char)
        required_bits = 4 + 8 + (len(self.data) * 8)
        required_bytes = (required_bits + 7) // 8  # Round up to nearest byte

        # Find the smallest version that can hold the data
        for version in range(1, self.MAX_VERSION + 1):
            if capacities[version - 1] >= required_bytes:
                return version

        raise ValueError(f'Data too large to fit in any QR code version with {self.error_correction} error correction')

    def get_mode_bits(self, mode, version: int) -> int:
        """Get the number of character count bits for a given mode and version."""
        from tokens import Mode as QRMode
        
        if version <= 9:
            counts = {
                QRMode.NUMERIC: 10,
                QRMode.ALPHANUMERIC: 9,
                QRMode.BYTE: 8,
                QRMode.KANJI: 8
            }
        elif version <= 26:
            counts = {
                QRMode.NUMERIC: 12,
                QRMode.ALPHANUMERIC: 11,
                QRMode.BYTE: 16,
                QRMode.KANJI: 10
            }
        else:  # version <= 40
            counts = {
                QRMode.NUMERIC: 14,
                QRMode.ALPHANUMERIC: 13,
                QRMode.BYTE: 16,
                QRMode.KANJI: 12
            }
        return counts[mode]

    def _add_version_information(self):
        """Add version information for QR codes version 7 and above"""
        if self.version < 7:
            return

        # Get version information bits
        version_bits = self.VERSION_INFO[self.version]

        # Place version information below bottom-left finder pattern
        for i in range(6):
            for j in range(3):
                bit = (version_bits >> (i * 3 + j)) & 1
                # Bottom left
                self.modules[self.size - 11 + j, i] = bit
                self.isfunction[self.size - 11 + j, i] = True
                # Top right
                self.modules[i, self.size - 11 + j] = bit
                self.isfunction[i, self.size - 11 + j] = True

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

def main():
    text = '677861663com.acme35584af52fa3-88d0-093b-6c14-b37ddafb59c528908608sg.com.dash.www0530329356521790265903SG.COM.NETS46968696003522G366948304B2AE13344004SG.SGQR209710339366720B439682.6366768027829126902859SG8236HELLO FOO2517Singapore3272B815'

    # text = 'Golden ratio = 1.61803398874983911374......'

    # Create QR code with debug output
    #print("\nGenerating QR code with debug output:")
    #qr_debug = QRCode(text, version=-1, error_correction='Q', mask=-1, debug=True)
    #qr_debug.draw()
    
    # Create QR code without debug output
    qr = QRCode(text, version=-1, error_correction='L', mask=-1, debug=True, mode="auto")
    qr.draw()

if __name__ == "__main__":
    main()