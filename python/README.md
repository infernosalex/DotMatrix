# QR Code Generator and Reader

A Python-based QR code generator and reader that can create QR codes from text and decode QR codes from images. This project implements QR code functionality from scratch, supporting various encoding modes and error correction levels.

## Features

- Generate QR codes from text input
- Read and decode QR codes from images
- Support for multiple encoding modes:
  - Numeric
  - Alphanumeric
  - Byte
  - Kanji
  - Auto (automatically selects the most efficient encoding, with support for optimal segmentation)
- Multiple error correction levels (L, M, Q, H)
- Automatic version selection based on data size
- Mask pattern optimization
- Visual QR code display using matplotlib

## Installation

1. Clone the repository:
```bash
git clone https://github.com/infernosalex/DotMatrix.git
cd python
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

The CLI is used through the `qr_tool.py` script, which provides two main commands:

### Creating a QR Code

To create a QR code from text:

```bash
python qr_tool.py create "Your text here"
```

This will display the generated QR code using matplotlib.

### Reading a QR Code

To read and decode a QR code from an image:

```bash
python qr_tool.py read path/to/qr/image.png
```

This will output the decoded text from the QR code.

### Help

To see all available options:

```bash
python qr_tool.py --help
```

## Project Structure

- `qr_tool.py`: Main command-line interface
- `qr_gen.py`: QR code generation implementation
- `tokens.py`: Tokenization of the input text for optimal encoding
- `qr_decode.py`: QR code decoding implementation
- `qr_image.py`: Image processing utilities
- `api.py`: API implementation

## Technical Details

The project implements the following key components:

- QR code matrix generation
- Reed-Solomon error correction
- Multiple encoding modes
- Mask pattern optimization
- Finder and alignment pattern placement
- Format and version information encoding
- Image processing for QR code reading

## Acknowledgments

This project is a from-scratch implementation of the QR code specification, following the ISO/IEC 18004:2015 standard. 