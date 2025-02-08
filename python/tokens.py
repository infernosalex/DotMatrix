from enum import Enum
import math

class Mode(Enum):
    NUMERIC = 0
    ALPHANUMERIC = 1
    BYTE = 2
    KANJI = 3

def get_mode_bits(mode: Mode, version: int) -> int:
    """Get the number of character count bits for a given mode and version."""
    if version <= 9:
        counts = {
            Mode.NUMERIC: 10,
            Mode.ALPHANUMERIC: 9,
            Mode.BYTE: 8,
            Mode.KANJI: 8
        }
    elif version <= 26:
        counts = {
            Mode.NUMERIC: 12,
            Mode.ALPHANUMERIC: 11,
            Mode.BYTE: 16,
            Mode.KANJI: 10
        }
    else:  # version <= 40
        counts = {
            Mode.NUMERIC: 14,
            Mode.ALPHANUMERIC: 13,
            Mode.BYTE: 16,
            Mode.KANJI: 12
        }
    return counts[mode]

def get_empty_segment_bits(mode: Mode, version: int) -> int:
    return 4 + get_mode_bits(mode, version)

def get_char_bit_lenght(mode: Mode, char: str) -> float:
    if mode == Mode.NUMERIC:
        return 20/6  # ~3.33 bits per digit
    elif mode == Mode.ALPHANUMERIC:
        return 33/6  # 5.5 bits per character
    elif mode == Mode.BYTE:
        # Calculate UTF-8 byte length * 8 bits per byte
        encoded = char.encode('utf-8')
        return len(encoded) * 8
    elif mode == Mode.KANJI:
        return 78/6  # 13 bits per character
    return float('inf')


def can_encode_char(mode: Mode, char: str) -> bool:
    """Check if a character can be encoded in a given mode."""
    if mode == Mode.NUMERIC:
        return char.isdigit()
    elif mode == Mode.ALPHANUMERIC:
        return char in '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ $%*+-./:'
    elif mode == Mode.BYTE:
        try:
            char.encode('utf-8')
            return True
        except UnicodeEncodeError:
            return False
    elif mode == Mode.KANJI:
        try:
            # For testing purposes, we'll consider any Unicode character above U+4E00 as Kanji
            # In a real implementation, you'd want to check against Shift JIS
            return ord(char) >= 0x4E00
        except:
            return False
    return False

def find_optimal_segmentation(text: str, version: int):
    if not text:
        return [], 0
    
    n = len(text)
    modes = list(Mode)

    f = {(i, m): float('inf') for i in range(n + 1) for m in modes}
    g = {(i, m): float('inf') for i in range(n + 1) for m in modes}
    h = {(i, m): None for i in range( n + 1) for m in modes}

    for m in modes:
        f[(0, m)] = 0
        g[(0, m)] = get_char_bit_lenght(m, text[0]) if can_encode_char(m, text[0]) else float('inf')


    for i in range(1,n+1):
        for curr_m in modes:
            if can_encode_char(curr_m, text[i-1]):
                for prev_m in modes:
                    if g[(i-1, prev_m)] == float('inf'):
                        continue

                    total_bits = g[(i-1, prev_m)]

                    if prev_m != curr_m:
                        total_bits = math.ceil(total_bits)
                        total_bits += get_empty_segment_bits(curr_m, version)

                    # print(f"{i} & {curr_m}:  {total_bits} = g{g[(i-1, prev_m)]} for {prev_m} + {get_empty_segment_bits(curr_m, version)}  ,     {text[i-1]}")

                    total_bits += get_char_bit_lenght(prev_m, text[i-1])

                    if total_bits < f[(i, curr_m)]:
                        f[(i, curr_m)] = total_bits
                        h[(i, curr_m)] = prev_m
            g[(i, curr_m)] = f[(i, curr_m)]

    best_bits = float('inf')
    best_mode = None
    
    for m in modes:
        if f[(n, m)] < best_bits:
            best_bits = f[(n, m)]
            best_mode = m

    # print(*f.items(), sep="\n\n")
    
    if best_bits == float('inf') or best_mode is None:
        raise ValueError("Unable to encode the input string with the given version")
    
    # Reconstruct optimal segmentation
    optimal_modes = [None] * (n+1)
    curr_mode = best_mode
    
    for i in range(n, 0, -1):
        optimal_modes[i-1] = curr_mode
        curr_mode = h[(i, curr_mode)]
        if curr_mode is None:
            raise ValueError("Invalid state in mode reconstruction")
    
    # Convert optimal modes into segments
    segments = []
    if n > 0:
        curr_segment = {'mode': optimal_modes[0], 'chars': [text[0]]}
        
        for i in range(1, n):
            if optimal_modes[i] == curr_segment['mode']:
                curr_segment['chars'].append(text[i])
            else:
                segments.append(curr_segment)
                curr_segment = {'mode': optimal_modes[i], 'chars': [text[i]]}
        
        segments.append(curr_segment)
    
    return segments, int(math.ceil(best_bits))

class Segment:
    def __init__(self, mode: Mode, data: str):
        self.mode = mode
        self.data = data

    def __str__(self):
        return f"Segment(mode={self.mode.name}, data='{self.data}')"

def optimize_segments(text: str, version: int) -> list[Segment]:
    """
    Convert text into an optimal list of segments for the given QR code version.
    Returns a list of Segment objects.
    """
    segments_data, total_bits = find_optimal_segmentation(text, version)
    return [Segment(seg['mode'], ''.join(seg['chars'])) for seg in segments_data]


# print(*optimize_segments("67128177921547861663com.acme35584af52fa3-88d0-093b-6c14-b37ddafb59c528908608sg.com.dash.www0530329356521790265903SG.COM.NETS46968696003522G33250183309051017567088693441243693268766948304B2AE13344004SG.SGQR209710339366720B439682.63667470805057501195235502733744600368027857918629797829126902859SG8236HELLO FOO2517Singapore3272B815", 1))
                        

    