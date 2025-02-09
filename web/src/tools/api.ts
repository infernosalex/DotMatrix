import axios from 'axios';

export interface QRDecodeResponse {
  intermediate_stages: {
    initial_matrix: number[][];
    unmasked_matrix: number[][];
    version: number;
    format_information: { error_correction: string, mask_pattern: number };
    version_information: string;
    data_bits_extracted: number[];
    data_capacity: number;
    data_bits_trimmed: number[];
    decoded_text: string;
  };
  qr_image_debug_logs: string;
  qr_decode_debug_logs: string;
}

export async function decodeQR(file: File): Promise<QRDecodeResponse> {
  const formData = new FormData();
  formData.append('image', file);
  const response = await axios.post('http://localhost:5000/api/decode', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
  return response.data;
}

export interface QRResponse {
  intermediate_stages: {
    after_temporary_format_bits: { modules: number[][], isfunction: boolean[][] };
    after_timing_patterns: { modules: number[][], isfunction: boolean[][] };
    after_finder_patterns: { modules: number[][], isfunction: boolean[][] };
    after_alignment_patterns: { modules: number[][], isfunction: boolean[][] };
    after_version_information: { modules: number[][], isfunction: boolean[][] };
    after_data_placement: { modules: number[][], isfunction: boolean[][] };
    final: { modules: number[][], isfunction: boolean[][] };
  };
  version: number;
  error_correction: string;
  mask: number;
  size: number;
  debug_logs: string;
}

export async function generateQR(text: string, error_correction: string, version: number, mask: number, mode: string): Promise<QRResponse> {
  const response = await axios.post('http://localhost:5000/api/generate', {
    data: text,
    error_correction,
    version,
    mask,
    mode
  });
  return response.data;
} 