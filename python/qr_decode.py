import numpy as np
from qr_gen import ReedSolomon

class QRDecoder:
    def __init__(self):
        # Format information bits and their corresponding error correction levels and mask patterns
        self.format_info_lookup = {
            0b000000000000000: ('M', 0), 0b000010100110111: ('M', 1), 0b000101001101110: ('M', 2),
            0b000111101011001: ('M', 3), 0b001000111101011: ('M', 4), 0b001010011011100: ('M', 5),
            0b001101110000101: ('M', 6), 0b001111010110010: ('M', 7), 0b010001111010110: ('L', 0),
            0b010011011100001: ('L', 1), 0b010100110111000: ('L', 2), 0b010110010001111: ('L', 3),
            0b011001000111101: ('L', 4), 0b011011100001010: ('L', 5), 0b011100001010011: ('L', 6),
            0b011110101100100: ('L', 7), 0b100001010011011: ('H', 0), 0b100011110101100: ('H', 1),
            0b100100011110101: ('H', 2), 0b100110111000010: ('H', 3), 0b101001101110000: ('H', 4),
            0b101011001000111: ('H', 5), 0b101100100011110: ('H', 6), 0b101110000101001: ('H', 7),
            0b110000101001111: ('Q', 0), 0b110010001111000: ('Q', 1), 0b110101100100001: ('Q', 2),
            0b110111000010110: ('Q', 3), 0b111000010100100: ('Q', 4), 0b111010110010011: ('Q', 5),
            0b111101011001010: ('Q', 6), 0b111111111111101: ('Q', 7)
        }
        # Format information mask pattern
        self.format_mask = 0b101010000010010
        # Mode indicator bits
        self.modes = {
            '0001': 'numeric',
            '0010': 'alphanumeric',
            '0100': 'byte',
            '1000': 'kanji'
        }
        # Character count indicator lengths for different versions and modes
        self.char_count_bits = {
            'numeric': {1: 10, 2: 10, 3: 10},
            'alphanumeric': {1: 9, 2: 9, 3: 9},
            'byte': {1: 8, 2: 8, 3: 8},
            'kanji': {1: 8, 2: 8, 3: 8}
        }

    def get_format_info(self, matrix):
        """Extract format information from the QR code matrix."""
        print("\n=== QR Code Decoding Debug ===")
        print("Step 1: Extracting Format Information")
        
        # Format info is stored in two locations
        # Location 1: Around the top-left finder pattern
        format_bits1 = []
        for i in range(6):
            format_bits1.append(matrix[8][i])  # Horizontal
        format_bits1.append(matrix[8][7])
        format_bits1.append(matrix[8][8])
        format_bits1.append(matrix[7][8])
        for i in range(5, -1, -1):
            format_bits1.append(matrix[i][8])  # Vertical

        # Location 2: Around the other finder patterns
        format_bits2 = []
        for i in range(7):
            format_bits2.append(matrix[matrix.shape[0]-1-i][8])  # Bottom-left vertical
        for i in range(8):
            format_bits2.append(matrix[8][matrix.shape[1]-8+i])  # Top-right horizontal

        # Convert bits to integer
        format_int1 = int(''.join(['1' if bit else '0' for bit in format_bits1]), 2)
        format_int2 = int(''.join(['1' if bit else '0' for bit in format_bits2]), 2)
        
        # XOR with mask pattern
        format_int1 ^= self.format_mask
        format_int2 ^= self.format_mask
        
        print(f"Format bits from location 1: {''.join(['1' if bit else '0' for bit in format_bits1])}")
        print(f"Format bits from location 2: {''.join(['1' if bit else '0' for bit in format_bits2])}")
        print(f"Format info after XOR 1: {bin(format_int1)[2:].zfill(15)}")
        print(f"Format info after XOR 2: {bin(format_int2)[2:].zfill(15)}")
        
        # Look up the format information
        if format_int1 in self.format_info_lookup:
            ec_level, mask_pattern = self.format_info_lookup[format_int1]
            print(f"Found format info: EC Level = {ec_level}, Mask Pattern = {mask_pattern}")
            return ec_level, mask_pattern
        elif format_int2 in self.format_info_lookup:
            ec_level, mask_pattern = self.format_info_lookup[format_int2]
            print(f"Found format info: EC Level = {ec_level}, Mask Pattern = {mask_pattern}")
            return ec_level, mask_pattern
        else:
            raise ValueError("Could not decode format information")

    def unmask_data(self, matrix, mask_pattern):
        """Apply the mask pattern to unmask the data."""
        print("\nStep 2: Unmasking Data")
        print(f"Applying mask pattern {mask_pattern}")
        
        unmasked = matrix.copy()
        rows, cols = matrix.shape
        
        for row in range(rows):
            for col in range(cols):
                # Skip the function patterns
                if self.is_function_pattern(row, col, cols):
                    continue
                
                # Apply the appropriate mask pattern
                mask_value = False
                if mask_pattern == 0:
                    mask_value = (row + col) % 2 == 0
                elif mask_pattern == 1:
                    mask_value = row % 2 == 0
                elif mask_pattern == 2:
                    mask_value = col % 3 == 0
                elif mask_pattern == 3:
                    mask_value = (row + col) % 3 == 0
                elif mask_pattern == 4:
                    mask_value = (row // 2 + col // 3) % 2 == 0
                elif mask_pattern == 5:
                    mask_value = ((row * col) % 2) + ((row * col) % 3) == 0
                elif mask_pattern == 6:
                    mask_value = (((row * col) % 2) + ((row * col) % 3)) % 2 == 0
                elif mask_pattern == 7:
                    mask_value = (((row + col) % 2) + ((row * col) % 3)) % 2 == 0
                
                if mask_value:
                    unmasked[row, col] = not matrix[row, col]
        
        print("Data unmasking complete")
        return unmasked

    def is_function_pattern(self, row, col, matrix_size):
        """Check if a position contains a function pattern."""
        # Finder patterns and their separators
        if row < 9 and col < 9:  # Top-left finder
            return True
        if row < 9 and col > matrix_size-9:  # Top-right finder
            return True
        if row > matrix_size-9 and col < 9:  # Bottom-left finder
            return True
        
        # Timing patterns
        if row == 6 or col == 6:
            return True
        
        # Alignment patterns for version 3
        if matrix_size == 29:  # Version 3
            alignment_centers = [22]  # Version 3 has one alignment pattern
            for center in alignment_centers:
                if abs(row - center) <= 2 and abs(col - center) <= 2:
                    return True
        
        return False

    def extract_data_bits(self, matrix):
        """Extract the data bits from the matrix in the correct zigzag pattern."""
        print("\nStep 3: Extracting Data Bits")
        rows, cols = matrix.shape
        data_bits = []
        
        # The zigzag pattern starts from the bottom right and moves upward
        # We'll move in columns of two, from right to left
        col = cols - 1
        up = True  # Direction flag
        
        while col >= 0:
            if col == 6:  # Skip timing pattern
                col -= 1
                continue
                
            # Process two columns at a time
            for row in range(rows) if not up else range(rows-1, -1, -1):
                # Process current column
                if not self.is_function_pattern(row, col, cols):
                    data_bits.append(matrix[row, col])
                
                # Process column - 1 if it exists
                if col > 0 and not self.is_function_pattern(row, col-1, cols):
                    data_bits.append(matrix[row, col-1])
            
            # Move to next column pair
            col -= 2
            up = not up  # Change direction
        
        print(f"Extracted {len(data_bits)} data bits")
        print("First 32 bits:", ''.join(['1' if bit else '0' for bit in data_bits[:32]]))
        
        # Group bits into bytes for debugging
        bytes_data = []
        for i in range(0, min(32, len(data_bits)), 8):
            byte_bits = data_bits[i:i+8]
            if len(byte_bits) == 8:
                byte = sum(bit << (7-j) for j, bit in enumerate(byte_bits))
                bytes_data.append(byte)
        print("First few bytes as hex:", [hex(b)[2:].zfill(2) for b in bytes_data])
        
        return data_bits

    def bits_to_bytes(self, bits):
        """Convert a list of bits to bytes."""
        bytes_data = []
        for i in range(0, len(bits), 8):
            if i + 8 <= len(bits):
                byte = 0
                for j in range(8):
                    byte = (byte << 1) | bits[i + j]
                bytes_data.append(byte)
        return bytes_data

    def parse_mode_and_count(self, data_bits, version):
        """Parse the mode indicator and character count from the data bits."""
        print("\nStep 4: Parsing Mode and Character Count")
        
        # Convert first 4 bits to mode indicator (MSB first)
        mode_bits = sum(bit << (3-i) for i, bit in enumerate(data_bits[:4]))
        mode_str = format(mode_bits, '04b')
        print(f"Mode indicator bits: {mode_str}")
        
        # For byte mode QR codes, the mode indicator should be 0100 (4)
        if mode_bits != 4:  # 0100 in binary
            raise ValueError(f"Invalid mode indicator: {mode_str}")
        
        mode = 'byte'  # We know it's byte mode
        print(f"Detected mode: {mode}")
        
        # Get character count indicator length for this version and mode
        count_length = self.char_count_bits[mode][version]
        
        # Get character count (MSB first)
        count_bits = data_bits[4:4+count_length]
        count = sum(bit << (count_length-1-i) for i, bit in enumerate(count_bits))
        print(f"Character count bits: {''.join(['1' if bit else '0' for bit in count_bits])}")
        print(f"Character count: {count}")
        
        return mode, count, 4 + count_length

    def decode_byte_mode(self, data_bits, char_count):
        """Decode data bits in byte mode."""
        print("\nStep 5: Decoding Byte Mode Data")
        
        bytes_data = []
        for i in range(0, char_count * 8, 8):
            if i + 8 <= len(data_bits):
                byte = sum(bit << (7-j) for j, bit in enumerate(data_bits[i:i+8]))
                bytes_data.append(byte)
        
        print(f"First few bytes as hex: {[hex(b)[2:].zfill(2) for b in bytes_data[:4]]}")
        print(f"Raw bytes: {bytes_data}")
        
        # Convert bytes to string
        try:
            text = bytes(bytes_data).decode('utf-8')
            print(f"Successfully decoded text: {text}")
            print("=== End QR Code Decoding Debug ===\n")
            return text
        except UnicodeDecodeError:
            print("Error: Failed to decode bytes as UTF-8")
            print("=== End QR Code Decoding Debug ===\n")
            return None

    def decode(self, matrix):
        """Decode a QR code matrix."""
        # Convert matrix to numpy array if it isn't already
        matrix = np.array(matrix)
        
        # Step 1: Get format information
        ec_level, mask_pattern = self.get_format_info(matrix)
        
        # Step 2: Unmask the data
        unmasked_matrix = self.unmask_data(matrix, mask_pattern)
        
        # Step 3: Extract the data bits
        data_bits = self.extract_data_bits(unmasked_matrix)
        
        # Step 4: Parse mode and character count
        # For this example, we know it's version 3
        mode, char_count, data_start = self.parse_mode_and_count(data_bits, version=3)
        
        # Step 5: Decode the data based on mode
        if mode == 'byte':
            return self.decode_byte_mode(data_bits[data_start:], char_count)
        else:
            raise ValueError(f"Unsupported mode: {mode}")

# Example usage
if __name__ == "__main__":
    from qr_gen import QRCode
    
    # Create a QR code
    text = "https://cs.unibuc.ro/~crusu/asc/"
    qr = QRCode(text, version=3, error_correction='L')
    qr.create_data_segment('byte')
    
    # Get the matrix
    matrix = qr.modules
    print(matrix)

    # print the matrix, with white and black squares
    for row in matrix:
        for element in row:
            print('██' if element else '  ', end='')
        print()
    
    # Decode the QR code
    decoder = QRDecoder()
    decoded_text = decoder.decode(matrix)
    print(f"Original text: {text}")
    print(f"Decoded text: {decoded_text}") 