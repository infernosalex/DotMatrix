# QR Code API Schema Documentation

## POST /api/generate

Generates a QR code with detailed intermediate stages of the generation process.

### Request

```json
{
    "data": "Hello World",              // Required: Text to encode in the QR code
    "version": -1,                      // Optional: QR version (1-40), -1 for auto-selection
    "error_correction": "L",            // Optional: Error correction level (L, M, Q, H)
    "mask": -1,                         // Optional: Mask pattern (0-7), -1 for auto-selection
    "mode": "auto"                      // Optional: Encoding mode (auto, numeric, alphanumeric, byte)
}
```

### Response

```json
{
    "intermediate_stages": {
        "after_temporary_format_bits": {
            "modules": [[0,0,0,...], ...],     // 2D array of module values
        },
        "after_timing_patterns": {
            "modules": [[0,0,0,...], ...],
        },

        "after_finder_patterns": {
            "modules": [[0,0,0,...], ...],
        },

        "after_alignment_patterns": {
            "modules": [[0,0,0,...], ...],
        },

        "after_version_information": {
            "modules": [[0,0,0,...], ...],
        },
        "data_segment": [...],              // Array of data bits
        "data_segment_with_ecc": [...],     // Array of data bits with error correction

        "after_data_placement": {
            "modules": [[0,0,0,...], ...],
        },

        "final": {
            "modules": [[0,0,0,...], ...],  // Final QR code matrix
        }
    },
    "version": 1,                          // Selected QR version
    "error_correction": "L",               // Selected error correction level
    "mask": 1,                            // Selected mask pattern
    "size": 21,                           // Size of QR code matrix
    "debug_logs": "..."                   // Debug output from generation process
}
```

## POST /api/decode

Decodes a QR code image and provides detailed information about the decoding process.

### Request

Multipart form data with:
- `image`: Image file containing the QR code (PNG format recommended)

### Response

```json
{
    "intermediate_stages": {
        "initial_matrix": [[0,0,0,...], ...],  // Initial binary matrix from image
        "version": 1,                          // Detected QR version
        "format_information": {
            "error_correction": "L",           // Detected error correction level
            "mask_pattern": 1                  // Detected mask pattern
        },
        "version_information": "...",          // Version information (versions >= 7)
        "unmasked_matrix": [[0,0,0,...], ...], // Matrix after unmasking
        "data_bits_extracted": [...],          // Raw extracted data bits
        "data_capacity": 100,                  // Data capacity in bits
        "data_bits_trimmed": [...],            // Trimmed data bits
        "decoded_text": "Hello World"          // Final decoded text
    },
    "qr_image_debug_logs": "...",             // Debug logs from image processing
    "qr_decode_debug_logs": "..."             // Debug logs from decoding process
}
```

Note: All arrays in the response containing numpy types are automatically converted to native Python types for JSON serialization. 