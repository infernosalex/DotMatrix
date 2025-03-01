import io
import contextlib
import tempfile
import os

from flask import Flask, request, jsonify, send_from_directory
import numpy as np
from flask_cors import CORS

# Import our QR modules
from qr_gen import QRCode
from qr_decode import QRDecode
from qr_image import image_to_matrix

# Initialize Flask app with static files configuration
app = Flask(__name__, 
            static_folder='../web/dist',  # Path to Vite build output
            static_url_path='')
CORS(app)

# Serve static files from Vite build
@app.route('/')
def serve_frontend():
    return send_from_directory(app.static_folder, 'index.html')

# Catch-all route to handle client-side routing
@app.route('/<path:path>')
def serve_static(path):
    if os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    return send_from_directory(app.static_folder, 'index.html')

# Helper function to convert numpy types to native Python types for JSON serialization
def convert_numpy(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    else:
        return obj

# Subclass to capture intermediate stages
class DebugQRCode(QRCode):
    def __init__(self, data, version, error_correction, mask, debug, mode):
        # Initialize basic parameters without calling parent's __init__
        self.data = data
        self.error_correction = error_correction
        self.mask = mask
        self.debug = debug
        self.mode = mode
        if version == -1:
            self.version = self._select_version()
        else:
            self.version = version
        self.size = 17 + 4 * self.version
        self.modules = np.zeros((self.size, self.size), dtype=int)
        self.isfunction = np.zeros((self.size, self.size), dtype=bool)
        self.intermediate = {}
        
        # Stage 1: Temporary format bits
        self._add_temporary_format_bits()
        self.intermediate['after_temporary_format_bits'] = {
            "modules": self.modules.tolist(),
            #"isfunction": self.isfunction.tolist()
        }
        
        # Stage 2: Timing patterns
        self._add_timing_patterns()
        self.intermediate['after_timing_patterns'] = {
            "modules": self.modules.tolist(),
            #"isfunction": self.isfunction.tolist()
        }
        
        # Stage 3: Finder patterns
        self._add_finder_patterns()
        self.intermediate['after_finder_patterns'] = {
            "modules": self.modules.tolist(),
            #"isfunction": self.isfunction.tolist()
        }
        
        # Stage 4: Alignment patterns
        self._add_alignment_patterns()
        self.intermediate['after_alignment_patterns'] = {
            "modules": self.modules.tolist(),
            #"isfunction": self.isfunction.tolist()
        }
        
        # Stage 5: Version information (if applicable)
        if self.version >= 7:
            self._add_version_information()
        self.intermediate['after_version_information'] = {
            "modules": self.modules.tolist(),
            #"isfunction": self.isfunction.tolist()
        }
        
        # Stage 6: Create data segment
        data_segment = self.create_data_segment(self.mode)
        self.intermediate['data_segment'] = data_segment
        
        # Stage 7: Add error correction
        data_segment_with_ecc = self.add_error_correction(data_segment)
        self.intermediate['data_segment_with_ecc'] = data_segment_with_ecc
        
        # Stage 8: Place data bits
        self._place_data_bits(data_segment_with_ecc)
        self.intermediate['after_data_placement'] = {
            "modules": self.modules.tolist(),
            #"isfunction": self.isfunction.tolist()
        }
        
        # Stage 9: Masking
        if self.mask == -1:
            self.select_mask()
        else:
            self._apply_mask(self.mask)
            self._draw_format_bits(self.mask)
        self.intermediate['final'] = {
            "modules": self.modules.tolist(),
            #"isfunction": self.isfunction.tolist()
        }


@app.route('/api/generate', methods=['POST'])
def generate_qr():
    # Parse JSON input
    data_payload = request.get_json()
    if not data_payload or 'data' not in data_payload:
        return jsonify({"error": "Missing 'data' field in JSON"}), 400

    text = data_payload.get('data')
    version = data_payload.get('version', -1)
    error_correction = data_payload.get('error_correction', 'L')
    mask = data_payload.get('mask', -1)
    mode = data_payload.get('mode', 'auto')

    # Capture debug logs during QR code generation
    debug_buffer = io.StringIO()
    with contextlib.redirect_stdout(debug_buffer):
        # Create DebugQRCode with debug=True
        qr = DebugQRCode(text, version=version, error_correction=error_correction, mask=mask, debug=True, mode=mode)
    debug_output = debug_buffer.getvalue()

    response = {
        "intermediate_stages": qr.intermediate,
        "version": qr.get_version(),
        "error_correction": qr.get_error_correction(),
        "mask": qr.get_mask(),
        "size": qr.get_size(),
        "debug_logs": debug_output
    }
    return jsonify(response)

# Subclass to capture intermediate stages during QR code decoding
class DebugQRDecode(QRDecode):
    def __init__(self, qr_code, debug=False, verbose=False):
        super().__init__(qr_code, debug=debug, verbose=verbose)
        self.intermediate = {}

    def decode(self) -> str:
        # Capture initial matrix state
        self.intermediate["initial_matrix"] = self.matrix.tolist()
        
        # Step 1: Determine the version of the QR code
        version = self.get_version()
        self.intermediate["version"] = version
        
        # Step 2: Extract format information (error correction level and mask pattern)
        error_correction, mask_pattern = self.extract_format_information()
        self.intermediate["format_information"] = {"error_correction": error_correction, "mask_pattern": mask_pattern}
        
        # Step 3: Extract version information for QR versions >= 7
        if version >= 7:
            ver_info = self.extract_version_information()
            self.intermediate["version_information"] = ver_info
        else:
            self.intermediate["version_information"] = None
        
        # Step 4: Unmask the QR code data
        self.unmask_data(mask_pattern)
        self.intermediate["unmasked_matrix"] = self.matrix.tolist()
        
        # Step 5: Extract data bits from the QR code
        data_bits = self.extract_data_bits()
        self.intermediate["data_bits_extracted"] = data_bits
        
        # Step 6: Record data capacity (if available)
        capacity = self._get_data_capacity() if hasattr(self, "_get_data_capacity") else len(data_bits)
        self.intermediate["data_capacity"] = capacity
        
        # Step 7: Apply error correction by extracting and correcting data blocks
        data_blocks, ec_blocks = self._extract_blocks(data_bits)
        corrected_data = self._apply_error_correction(data_blocks, ec_blocks)
        
        # Convert corrected data bytes back into a bit stream
        corrected_bits = []
        for byte in corrected_data:
            bits = [(byte >> i) & 1 for i in range(7, -1, -1)]
            corrected_bits.extend(bits)
        self.intermediate["data_bits_trimmed"] = corrected_bits
        
        # Step 8: Decode the final message from the corrected bit stream
        decoded_text = self.decode_data(corrected_bits)
        self.intermediate["decoded_text"] = decoded_text
        
        return decoded_text

# Update the decode_qr endpoint to use DebugQRDecode
@app.route('/api/decode', methods=['POST'])
def decode_qr():
    if 'image' not in request.files:
        return jsonify({"error": "Missing image file"}), 400
    image_file = request.files['image']

    # Save the uploaded file to a temporary file on disk
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
        image_file.save(tmp.name)
        tmp_file_path = tmp.name

    # Capture debug logs during image to matrix conversion
    image_debug_buffer = io.StringIO()
    with contextlib.redirect_stdout(image_debug_buffer):
        binary_matrix = image_to_matrix(tmp_file_path, threshold=128, debug=True)
    image_debug_output = image_debug_buffer.getvalue()
    binary_matrix_list = binary_matrix.tolist() if hasattr(binary_matrix, 'tolist') else binary_matrix

    # Capture debug logs during QR decoding
    decode_debug_buffer = io.StringIO()
    with contextlib.redirect_stdout(decode_debug_buffer):
        qr_decoder = DebugQRDecode(binary_matrix, debug=True)
        decoded_text = qr_decoder.decode()
    decode_debug_output = decode_debug_buffer.getvalue()

    response = {
        "intermediate_stages": qr_decoder.intermediate,
        "qr_image_debug_logs": image_debug_output,
        "qr_decode_debug_logs": decode_debug_output
    }
    response = convert_numpy(response)  # Convert any numpy types to native Python types
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True) 