import numpy as np
from qr_gen import QRCode


class QRDecode:
    def __init__(self, qr_code: np.ndarray, debug: bool = False, verbose: bool = False) -> None:
        self.matrix = qr_code
        self.debug = debug
        self.verbose = verbose

        # Validate input matrix
        if not qr_code.any() or any(len(row) != len(self.matrix) for row in self.matrix):
            raise ValueError("Invalid QR code matrix")

    def get_version(self) -> int:
        """Get the QR code version based on matrix size."""
        size = len(self.matrix)
        version = (size - 17) // 4
        if self.debug:
            print(f"[DEBUG] get_version: Matrix size = {size}, computed version = {version}")
            print(f"[DEBUG] get_version: Expected matrix size for this version = {17 + 4 * version}")
        return version

    def mark_function_modules(self) -> np.ndarray:
        """Mark all function modules in the QR code."""
        size = len(self.matrix)
        function_modules = np.zeros((size, size), dtype=bool)
        
        # Compute version before using it
        version = self.get_version()

        # Helper function to add a finder pattern and its separator at given position
        def add_finder_pattern(row, col):
            # Add the 7x7 finder pattern
            for r in range(7):
                for c in range(7):
                    if 0 <= row + r < size and 0 <= col + c < size:
                        function_modules[row + r, col + c] = True
            
            # Add separator (white border)
            # Mark the entire 8x8 region around the finder pattern
            for r in range(-1, 8):
                for c in range(-1, 8):
                    if 0 <= row + r < size and 0 <= col + c < size:
                        # Only mark if it's not part of the finder pattern itself
                        if r < 0 or r >= 7 or c < 0 or c >= 7:
                            function_modules[row + r, col + c] = True

        # Add finder patterns at the three corners
        add_finder_pattern(0, 0)        # Top-left
        add_finder_pattern(0, size - 7)   # Top-right
        add_finder_pattern(size - 7, 0)   # Bottom-left

        # Add timing patterns
        for i in range(size):
            function_modules[6, i] = True  # Horizontal timing pattern
            function_modules[i, 6] = True  # Vertical timing pattern

        # Add format information areas
        # Around top-left finder pattern
        for i in range(9):
            if i != 6:  # Skip timing pattern
                function_modules[8, i] = True  # Horizontal format info
                if i < 8:
                    function_modules[i, 8] = True  # Vertical format info

        # Around top-right finder pattern
        for i in range(8):
            function_modules[8, size - 1 - i] = True  # Horizontal format info

        # Around bottom-left finder pattern
        for i in range(8):
            function_modules[size - 1 - i, 8] = True  # Vertical format info

        # Dark module
        function_modules[size - 8, 8] = True

        # Mark alignment patterns for version >= 2
        if version >= 2:
            centers = self._get_alignment_pattern_centers()
            for r in centers:
                for c in centers:
                    # Skip the finder pattern regions
                    if (r, c) in [(6, 6), (6, size - 7), (size - 7, 6)]:
                        continue
                    for i in range(-2, 3):
                        for j in range(-2, 3):
                            if 0 <= r + i < size and 0 <= c + j < size:
                                function_modules[r + i, c + j] = True

        # Add version information areas for version >= 7
        if version >= 7:
            # Add version information below bottom-left finder pattern
            for i in range(6):
                for j in range(3):
                    function_modules[size - 11 + j, i] = True  # Bottom-left version info
                    function_modules[i, size - 11 + j] = True  # Top-right version info

        if self.verbose:
            print("[DEBUG] mark_function_modules: Function modules matrix:")
            for i in range(size):
                row = ""
                for j in range(size):
                    row += "██" if function_modules[i, j] else "  "
                print(row)

        return function_modules

    def _get_alignment_pattern_centers(self) -> list:
        """Return a list of alignment pattern center positions for the QR code.
        Uses a lookup table for versions 2 to 10.
        """
        version = self.get_version()
        if version == 1:
            return []
        lookup = {
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
        if version in lookup:
            return lookup[version]
        else:
            end = len(self.matrix) - 7
            count = (version // 7) + 2
            if count == 2:
                return [6, end]
            step = (end - 6) // (count - 1)
            centers = [6] + [6 + i * step for i in range(1, count - 1)] + [end]
            return centers

    def extract_format_information(self) -> tuple[str, int]:
        """Extract format information from the QR code."""
        
        first_copy = 0
        for col in [0, 1, 2, 3, 4, 5, 7, 8]:
            first_copy = (first_copy << 1) | (self.matrix[8][col] & 1)
            if self.verbose:
                print(f"[DEBUG] extract_format_information: first_copy after col {col} = {first_copy:0b}")

        second_copy = 0
        for row in [7, 5, 4, 3, 2, 1, 0]:
            second_copy = (second_copy << 1) | (self.matrix[row][8] & 1)
            if self.verbose:
                print(f"[DEBUG] extract_format_information: second_copy after row {row} = {second_copy:0b}")

        raw_format = (first_copy << 7) | second_copy
        if self.verbose:
            print(f"[DEBUG] extract_format_information: Raw format (before unmask) = {raw_format:015b}")

        # Unmask the format bits using the mask 0x5412
        format_info = raw_format ^ 0x5412
        if self.verbose:
            print(f"[DEBUG] extract_format_information: Format info (after unmask) = {format_info:015b}")

        # Extract the 5 format data bits (bits 14 to 10) as per QR spec
        data_bits = (format_info >> 10) & 0b11111
        if self.debug:
            print(f"[DEBUG] extract_format_information: Data bits = {data_bits:05b}")

        # The first two bits (bits 4-3) are error correction level, and the last three bits (bits 2-0) are mask pattern
        error_correction = {
            0b00: 'M',
            0b01: 'L',
            0b10: 'H',
            0b11: 'Q'
        }[(data_bits >> 3) & 0b11]
        mask_pattern = data_bits & 0b111
        if self.debug:
            print(f"[DEBUG] extract_format_information: Extracted error correction = {error_correction}, mask pattern = {mask_pattern}")

        return error_correction, mask_pattern

    def extract_version_information(self) -> int:
        """Extract version information from the QR code (for version >= 7)."""
        # For versions less than 7, just return the computed version
        version = self.get_version()
        if version < 7:
            return version

        size = len(self.matrix)
        raw_version = 0

        # Extract version information from bottom-left block
        for r in range(size - 11, size - 8):
            for c in range(5, -1, -1):
                raw_version = (raw_version << 1) | (self.matrix[r][c] & 1)

        if self.debug:
            print(f"[DEBUG] extract_version_information: Raw version bits = {raw_version:018b}")

        corrected_version = self.correct_version_information(raw_version)
        return corrected_version

    def _compute_valid_version_info(self, version: int) -> int:
        """Compute the valid 18-bit version information codeword for the given version using BCH error correction."""
        generator = 0x1F25
        codeword = version << 12
        for i in range(17, 11, -1):
            if codeword & (1 << i):
                codeword ^= generator << (i - 12)
        valid_codeword = (version << 12) | (codeword & 0xFFF)
        if self.debug:
            print(f"[DEBUG] _compute_valid_version_info: version = {version}, valid_codeword = {valid_codeword:018b}")
        return valid_codeword

    def correct_version_information(self, raw_version: int) -> int:
        """Correct errors in raw version information bits using BCH code by brute-force matching valid codewords."""
        if self.debug:
            print(f"[DEBUG] correct_version_information: raw_version = {raw_version:018b}")
        for version in range(7, 41):
            valid_codeword = self._compute_valid_version_info(version)
            diff = bin(valid_codeword ^ raw_version).count('1')
            if self.debug:
                print(f"[DEBUG] correct_version_information: testing version {version} with diff = {diff}")
            if diff <= 3:
                if self.debug:
                    print(f"[DEBUG] correct_version_information: corrected to version {version}")
                return version
        raise ValueError("Failed to correct version information")

    def unmask_data(self, mask_pattern: int) -> None:
        """Unmask the data in the QR code by inverting bits in data modules as per the mask pattern."""
        size = len(self.matrix)
        function_modules = self.mark_function_modules()

        def mask_condition(i: int, j: int, mask: int) -> bool:
            if mask == 0:
                return (i + j) % 2 == 0
            elif mask == 1:
                return i % 2 == 0
            elif mask == 2:
                return j % 3 == 0
            elif mask == 3:
                return (i + j) % 3 == 0
            elif mask == 4:
                return ((i // 2) + (j // 3)) % 2 == 0
            elif mask == 5:
                return ((i * j) % 2 + (i * j) % 3) == 0
            elif mask == 6:
                return (((i * j) % 2 + (i * j) % 3) % 2) == 0
            elif mask == 7:
                return (((i + j) % 2 + (i * j) % 3) % 2) == 0
            else:
                raise ValueError(f"Invalid mask pattern: {mask}")

        for i in range(size):
            for j in range(size):
                # Only unmask data modules
                if not function_modules[i, j]:
                    if mask_condition(i, j, mask_pattern):
                        self.matrix[i][j] ^= 1
        if self.verbose:
            print(f"[DEBUG] unmask_data: Finished unmasking process. Matrix:")
            for row in self.matrix:
                print("".join("██" if bit == 1 else "  " for bit in row))
            print("[DEBUG] unmask_data: Finished unmasking process.")
        # Ensure matrix remains binary
        assert np.all((self.matrix == 0) | (self.matrix == 1)), "Matrix contains non-binary values after unmasking"

    def extract_data_bits(self) -> list[int]:
        """Extract the data bits from the QR code following the standard zigzag order."""
        if self.debug:
            print("[DEBUG] extract_data_bits: Starting extraction of data bits.")
        # Mark function modules to know which modules are reserved
        function_modules = self.mark_function_modules()
        n = len(self.matrix)
        data_bits = []

        col = n - 1
        upward = True
        while col > 0:
            # Skip the vertical timing pattern column
            if col == 6:
                col -= 1
            # Determine the row order based on scanning direction
            row_range = range(n-1, -1, -1) if upward else range(0, n)
            for row in row_range:
                for c in [col, col - 1]:
                    if c < 0 or c >= n:
                        continue
                    # Only extract the bit if it's not part of a function module
                    if not function_modules[row, c]:
                        bit = int(self.matrix[row][c])
                        data_bits.append(bit)
                        if self.verbose:
                            print(f"[DEBUG] extract_data_bits: Appended bit {bit} from position ({row},{c}).")
            upward = not upward
            col -= 2
        if self.debug:
            print(f"[DEBUG] extract_data_bits: Finished extraction, total bits = {len(data_bits)}")
        # Assert that all extracted bits are binary
        assert all(bit in (0, 1) for bit in data_bits), "Matrix contains non-binary values in data bits"
        return data_bits

    def decode_data(self, data_bits: list[int]) -> str:
        """Decode the data bits and return the message, supporting multiple segmentation modes."""
        pointer = 0
        result = ""
        version = self.get_version()
        if self.debug:
            print("[DEBUG] decode_data: Starting multi-segment decoding.")
        while pointer + 4 <= len(data_bits):
            mode_indicator = int(''.join(str(bit) for bit in data_bits[pointer:pointer+4]), 2)
            if self.debug:
                print(f"[DEBUG] decode_data: Mode indicator = {mode_indicator:04b} at pointer {pointer}")
            pointer += 4
            if mode_indicator == 0:  # Terminator
                if self.debug:
                    print("[DEBUG] decode_data: Encountered terminator mode indicator")
                break
            if mode_indicator == 4:  # Byte mode
                count_indicator_length = 8 if version < 10 else 16
                if pointer + count_indicator_length > len(data_bits):
                    if self.debug:
                        print("[DEBUG] decode_data: Not enough bits for byte mode character count indicator")
                    break
                count = int(''.join(str(bit) for bit in data_bits[pointer:pointer+count_indicator_length]), 2)
                if self.debug:
                    print(f"[DEBUG] decode_data: Byte mode, character count = {count}")
                pointer += count_indicator_length
                for i in range(count):
                    if pointer + 8 > len(data_bits):
                        if self.debug:
                            print(f"[DEBUG] decode_data: Not enough bits for byte {i+1}/{count}")
                        break
                    byte_val = int(''.join(str(bit) for bit in data_bits[pointer:pointer+8]), 2)
                    if self.verbose:
                        print(f"[DEBUG] decode_data: Extracted byte value {byte_val} from bits at pointer {pointer}")
                    result += chr(byte_val)
                    pointer += 8
            elif mode_indicator == 1:  # Numeric mode
                count_indicator_length = 10 if version < 10 else (12 if version < 27 else 14)
                if pointer + count_indicator_length > len(data_bits):
                    if self.debug:
                        print("[DEBUG] decode_data: Not enough bits for numeric mode character count indicator")
                    break
                count = int(''.join(str(bit) for bit in data_bits[pointer:pointer+count_indicator_length]), 2)
                if self.debug:
                    print(f"[DEBUG] decode_data: Numeric mode, digit count = {count}")
                pointer += count_indicator_length
                # Process groups of 3 digits
                while count >= 3:
                    if pointer + 10 > len(data_bits):
                        if self.debug:
                            print("[DEBUG] decode_data: Not enough bits for a group of 3 digits in numeric mode")
                        break
                    group_val = int(''.join(str(bit) for bit in data_bits[pointer:pointer+10]), 2)
                    if self.debug:
                        print(f"[DEBUG] decode_data: Numeric mode: Extracted group value {group_val} from bits at pointer {pointer}")
                    result += f"{group_val:03d}"
                    pointer += 10
                    count -= 3
                if count == 2:
                    if pointer + 7 > len(data_bits):
                        if self.debug:
                            print("[DEBUG] decode_data: Not enough bits for a group of 2 digits in numeric mode")
                        break
                    group_val = int(''.join(str(bit) for bit in data_bits[pointer:pointer+7]), 2)
                    if self.debug:
                        print(f"[DEBUG] decode_data: Numeric mode: Extracted 2-digit group value {group_val} from bits at pointer {pointer}")
                    result += f"{group_val:02d}"
                    pointer += 7
                elif count == 1:
                    if pointer + 4 > len(data_bits):
                        if self.debug:
                            print("[DEBUG] decode_data: Not enough bits for a single digit in numeric mode")
                        break
                    group_val = int(''.join(str(bit) for bit in data_bits[pointer:pointer+4]), 2)
                    if self.debug:
                        print(f"[DEBUG] decode_data: Numeric mode: Extracted single digit value {group_val} from bits at pointer {pointer}")
                    result += f"{group_val:d}"
                    pointer += 4
            elif mode_indicator == 2:  # Alphanumeric mode
                count_indicator_length = 9 if version < 10 else (11 if version < 27 else 13)
                if pointer + count_indicator_length > len(data_bits):
                    if self.debug:
                        print("[DEBUG] decode_data: Not enough bits for alphanumeric mode character count indicator")
                    break
                count = int(''.join(str(bit) for bit in data_bits[pointer:pointer+count_indicator_length]), 2)
                if self.debug:
                    print(f"[DEBUG] decode_data: Alphanumeric mode, character count = {count}")
                pointer += count_indicator_length
                alphanum_table = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:"
                while count >= 2:
                    if pointer + 11 > len(data_bits):
                        if self.debug:
                            print("[DEBUG] decode_data: Not enough bits for a pair in alphanumeric mode")
                        break
                    value = int(''.join(str(bit) for bit in data_bits[pointer:pointer+11]), 2)
                    if self.verbose:
                        print(f"[DEBUG] decode_data: Alphanumeric mode: Extracted pair value {value} from bits at pointer {pointer}")
                    if self.debug:
                        print(result, " add ", value, " = ", alphanum_table[value // 45], alphanum_table[value % 45])
                    result += alphanum_table[value // 45] + alphanum_table[value % 45]
                    pointer += 11
                    count -= 2
                if count == 1:
                    if pointer + 6 > len(data_bits):
                        if self.debug:
                            print("[DEBUG] decode_data: Not enough bits for a single character in alphanumeric mode")
                        break
                    value = int(''.join(str(bit) for bit in data_bits[pointer:pointer+6]), 2)
                    if self.verbose:
                        print(f"[DEBUG] decode_data: Alphanumeric mode: Extracted single character value {value} from bits at pointer {pointer}")
                    result += alphanum_table[value]
                    pointer += 6
            else:
                if self.debug:
                    print(f"[DEBUG] decode_data: Unsupported mode indicator {mode_indicator:04b}")
                break
        if self.debug:
            print(f"[DEBUG] decode_data: Final decoded message = {result}")
        return result

    def _get_data_capacity(self) -> int:
        """Return the number of data bits (not including error correction) for the QR code.
        Uses a lookup table for versions 1 to 4.
        """
        version = self.get_version()
        error_correction, _ = self.extract_format_information()
        lookup = {
            1: {'L': 152, 'M': 128, 'Q': 104, 'H': 72},
            2: {'L': 272, 'M': 224, 'Q': 176, 'H': 128},
            3: {'L': 440, 'M': 352, 'Q': 272, 'H': 208},
            4: {'L': 640, 'M': 512, 'Q': 384, 'H': 288}
        }
        if version in lookup:
            return lookup[version].get(error_correction, 152)
        else:
            # Fallback for higher versions
            total_modules = len(self.matrix) ** 2
            function_modules = np.sum(self.mark_function_modules())
            return int(total_modules - function_modules)

    def decode(self) -> str:
        """Decode the QR code and return the message."""
        # 1. Get version
        version = self.get_version()
        if self.debug:
            print(f"[DEBUG] decode: QR code version = {version}")

        # 2. Extract format information
        error_correction, mask_pattern = self.extract_format_information()
        if self.debug:
            print(f"[DEBUG] decode: Error correction level = {error_correction}, Mask pattern = {mask_pattern}")

        # 3. Extract version information (for version >= 7)
        if version >= 7:
            version_info = self.extract_version_information()
            if self.debug:
                print(f"[DEBUG] decode: Version information = {version_info}")

        # 4. Unmask data
        self.unmask_data(mask_pattern)
        if self.debug:
            print("[DEBUG] decode: Data unmasked")

        # 5. Extract data bits and trim extra bits
        data_bits = self.extract_data_bits()
        capacity = self._get_data_capacity()
        if self.debug:
            print(f"[DEBUG] decode: Trimming data bits to capacity: {capacity} bits (out of {len(data_bits)})")
        data_bits = data_bits[:capacity]
        if self.debug:
            print(f"[DEBUG] decode: Using {len(data_bits)} data bits for decoding")

        # 6. Decode data bits
        return self.decode_data(data_bits)


if __name__ == "__main__":
    #qr_code = QRCode('https://cs.unibuc.ro/~crusu/asc/index.html', version=-1, error_correction='H', mask=-1, debug=False, mode="byte")
    # qr_code.draw()
    qr_code = QRCode('https://cs.unibuc.ro/~crusu/asc/index.html', version=-1, error_correction='L', mask=-1, debug=False, mode="auto")
    data_bits = qr_code.get_matrix()
    version = qr_code.get_version()
    error_correction = qr_code.get_error_correction()
    mask_pattern = qr_code.get_mask()
    
    print("Generated QR Code:")
    for row in data_bits:
        print("".join("██" if bit == 1 else "  " for bit in row))
    print(f"Generated version: {version}")
    print(f"Error correction: {error_correction}")
    print(f"Mask pattern: {mask_pattern}")


    qr_decode = QRDecode(data_bits, debug=True)
    decoded_text = qr_decode.decode()
    print(f"\nDecoded text: {decoded_text}")

