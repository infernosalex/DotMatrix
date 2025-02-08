import argparse
import sys

from qr_gen import QRCode
from qr_decode import QRDecode
from qr_image import image_to_matrix


def create_qr(text: str) -> None:
    """Create a QR code from the provided text and display it using qr.draw."""
    # All debug/verbose off, use auto mode
    qr = QRCode(text, version=-1, error_correction='L', mask=-1, debug=False, mode='auto')
    qr.draw()


def read_qr(image_path: str) -> str:
    """Read a QR code from an image file, decode it, and return the decoded string."""
    # Convert image to binary matrix with default threshold, no debug
    matrix = image_to_matrix(image_path, threshold=128, debug=False)
    decoder = QRDecode(matrix, debug=False, verbose=False)
    result = decoder.decode()
    print("Decoded text:", result)
    return result


def main():
    parser = argparse.ArgumentParser(description='QR Tool: Create or read QR codes.')
    subparsers = parser.add_subparsers(dest='command', required=True, help='Sub-command to run')

    # Subcommand for creating a QR code
    create_parser = subparsers.add_parser('create', help='Create a QR code from text and display it.')
    create_parser.add_argument('text', type=str, help='Text to encode in the QR code')

    # Subcommand for reading a QR code from an image
    read_parser = subparsers.add_parser('read', help='Read and decode a QR code from an image file.')
    read_parser.add_argument('file', type=str, help='Path to the QR code image file')

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)

    args = parser.parse_args()

    if args.command == 'create':
        create_qr(args.text)
    elif args.command == 'read':
        read_qr(args.file)
    else:
        parser.print_help()


if __name__ == '__main__':
    main() 