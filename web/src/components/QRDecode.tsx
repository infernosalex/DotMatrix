import { useState, ChangeEvent } from 'react';
import ModuleCell from './common/ModuleCell';
import PrimaryButton from './common/PrimaryButton';
import IconButton from './common/IconButton';
import StageButton from './common/StageButton';
import { decodeQR as callDecodeQR } from '../tools/api';
import { FaLongArrowAltLeft, FaLongArrowAltRight } from 'react-icons/fa';
import CollapsibleSection from './common/CollapsibleSection';

// Define the interface for the decode response
interface QRDecodeResponse {
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

export function QRDecode() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [decodeData, setDecodeData] = useState<QRDecodeResponse | null>(null);
  const [isDecoding, setIsDecoding] = useState(false);
  const [currentStage, setCurrentStage] = useState(0);

  // Define stages for matrix display
  const stages = ['Initial Matrix', 'Unmasked Matrix', 'Intermediate Info', 'Decoded Text'];

  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0]);
    }
  };

  const decodeQR = async () => {
    if (!selectedFile) return;
    setIsDecoding(true);
    setDecodeData(null);
    setCurrentStage(0);

    try {
      const data = await callDecodeQR(selectedFile);
      setDecodeData(data);
    } catch (error) {
      console.error('Error decoding QR code:', error);
    } finally {
      setIsDecoding(false);
    }
  };

  // Add helper functions to render the stage content
  const renderMatrix = (matrix: number[][]) => {
    if (!matrix || matrix.length === 0 || matrix[0].length === 0) {
      return <p className="text-center text-gray-700">Matrix data not available</p>;
    }
    return (
      <div
        className="grid gap-0 bg-white p-4 rounded-lg mt-4 aspect-square"
        style={{
          gridTemplateColumns: `repeat(${matrix[0].length}, 1fr)`,
          gridTemplateRows: `repeat(${matrix.length}, 1fr)`
        }}
      >
        {matrix.map((row, i) =>
          row.map((cell, j) => (
            <div key={`${i}-${j}`} className="aspect-square">
              <ModuleCell cell={!!cell} />
            </div>
          ))
        )}
      </div>
    );
  };

  const getStageContent = () => {
    if (!decodeData) return null;
    switch (currentStage) {
      case 0:
        return renderMatrix(decodeData.intermediate_stages.initial_matrix);
      case 1:
        return renderMatrix(decodeData.intermediate_stages.unmasked_matrix);
      case 2:
        return (
          <div className="p-4 space-y-2">
            <div><strong>QR Version:</strong> {decodeData.intermediate_stages.version}</div>
            <div><strong>Error Correction:</strong> {decodeData.intermediate_stages.format_information.error_correction}</div>
            <div><strong>Mask Pattern:</strong> {decodeData.intermediate_stages.format_information.mask_pattern}</div>
            <div><strong>Version Information:</strong> {decodeData.intermediate_stages.version_information}</div>
            <div><strong>Data Capacity:</strong> {decodeData.intermediate_stages.data_capacity}</div>
            <div>
              <strong>Data Bits Extracted:</strong>
              <div className="bg-gray-100 p-2 rounded break-words">{decodeData.intermediate_stages.data_bits_extracted}</div>
            </div>
            <div>
              <strong>Data Bits Trimmed:</strong>
              <div className="bg-gray-100 p-2 rounded break-words">{decodeData.intermediate_stages.data_bits_trimmed}</div>
            </div>
          </div>

        );
      case 3:
        return (
          <div className="p-4">
            <p className="text-xl font-semibold mb-2">Decoded Text:</p>
            <p className="text-lg break-words">{decodeData.intermediate_stages.decoded_text}</p>
          </div>

        );
      default:
        return null;
    }
  };

  return (
    <div className="max-w-5xl mx-auto flex flex-col gap-y-4 mt-8">
      <div className="bg-white rounded-xl p-6 mb-4 shadow-xl">
        <h2 className="text-2xl font-bold mb-6 text-center text-gray-900">Decode QR Code</h2>
        <div className="flex flex-col gap-4 max-w-2xl mx-auto">
          <div>
            <label className="block text-sm font-medium text-gray-900 mb-1">Select QR Code Image</label>
            <input 
              type="file" 
              accept="image/*"
              onChange={handleFileChange}
              className="w-full p-0 file:p-3 file:mr-4 bg-gray-50 border border-gray-300 rounded-lg text-gray-900 file:bg-gray-200 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition duration-150 ease-in-out shadow-sm hover:shadow-md"
            />
          </div>
          <PrimaryButton onClick={decodeQR} disabled={isDecoding || !selectedFile}>
            {isDecoding ? 'Decoding...' : 'Decode QR'}
          </PrimaryButton>
        </div>
      </div>

      {decodeData && (
        <>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-white rounded-xl p-6 shadow-xl h-fit">
              <h3 className="text-xl font-semibold mb-6 text-center text-gray-900">Matrix Stages</h3>
              <div className="flex flex-col gap-y-1">
                {stages.map((stage, index) => (
                  <StageButton
                    key={stage}
                    active={currentStage === index}
                    isLast={index === stages.length - 1}
                    onClick={() => setCurrentStage(index)}
                  >
                    {stage}
                  </StageButton>
                ))}
              </div>
            </div>

            <div className="bg-white rounded-xl p-6 shadow-xl">
              <div className="flex items-center justify-between mb-6">
                <IconButton
                  onClick={() => currentStage > 0 && setCurrentStage(currentStage - 1)}
                  disabled={currentStage === 0}
                >
                  <FaLongArrowAltLeft />
                </IconButton>
                <h3 className="text-xl font-semibold text-center text-gray-900">
                  {stages[currentStage]}
                </h3>
                <IconButton
                  onClick={() => currentStage < stages.length - 1 && setCurrentStage(currentStage + 1)}
                  disabled={currentStage === stages.length - 1}
                >
                  <FaLongArrowAltRight />
                </IconButton>
              </div>
              {getStageContent()}
            </div>
          </div>

          <CollapsibleSection title="Debug Logs">
            <div>
              <h4 className="font-bold mb-1">QR Image Debug Logs</h4>
              <pre className="text-xs font-mono overflow-x-auto bg-gray-100 p-2 rounded">{decodeData.qr_image_debug_logs}</pre>
            </div>
            <div>
              <h4 className="font-bold mb-1">QR Decode Debug Logs</h4>
              <pre className="text-xs font-mono overflow-x-auto bg-gray-100 p-2 rounded">{decodeData.qr_decode_debug_logs}</pre>
            </div>
          </CollapsibleSection>
        </>
      )}
    </div>
  );
} 