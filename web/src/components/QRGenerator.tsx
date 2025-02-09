import { useState } from 'react'
import ModuleCell from './common/ModuleCell';
import PrimaryButton from './common/PrimaryButton';
import IconButton from './common/IconButton';
import StageButton from './common/StageButton';
import axios from 'axios'
import { FaLongArrowAltLeft, FaLongArrowAltRight } from "react-icons/fa";
import CollapsibleSection from './common/CollapsibleSection';

interface QRResponse {
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

export function QRGenerator() {
  const [text, setText] = useState('');
  const [currentStage, setCurrentStage] = useState(0);
  const [qrData, setQrData] = useState<QRResponse | null>(null);
  const [isGenerating, setIsGenerating] = useState(false);
  const [advancedOptionsVisible, setAdvancedOptionsVisible] = useState(false);
  const [version, setVersion] = useState<number>(-1);
  const [errorCorrection, setErrorCorrection] = useState('L');
  const [mask, setMask] = useState<number>(-1);
  const [mode, setMode] = useState('auto');

  const stages = [
    'Format Bits',
    'Timing Patterns',
    'Finder Patterns',
    'Alignment Patterns',
    'Version Information',
    'Data Placement',
    'Final QR Code'
  ];

  const generateQR = async () => {
    try {
      setIsGenerating(true);
      setCurrentStage(0);
      
      const response = await axios.post('http://localhost:5000/api/generate', {
        data: text,
        error_correction: errorCorrection,
        version: version,
        mask: mask,
        mode: mode
      });

      setQrData(response.data);
    } catch (error) {
      console.error('Error generating QR code:', error);
    } finally {
      setIsGenerating(false);
    }
  };

  const getMatrixForStage = (stage: number) => {
    if (!qrData) return null;
    switch (stage) {
      case 0:
        return qrData.intermediate_stages.after_temporary_format_bits;
      case 1:
        return qrData.intermediate_stages.after_timing_patterns;
      case 2:
        return qrData.intermediate_stages.after_finder_patterns;
      case 3:
        return qrData.intermediate_stages.after_alignment_patterns;
      case 4:
        return qrData.intermediate_stages.after_version_information;
      case 5:
        return qrData.intermediate_stages.after_data_placement;
      case 6:
        return qrData.intermediate_stages.final;
      default:
        return null;
    }
  };

  const currentMatrix = getMatrixForStage(currentStage);

  return (
    <div className="max-w-5xl mx-auto flex flex-col gap-y-4">
      <div className="bg-white rounded-xl p-6 mb-4 shadow-xl">
        <h2 className="text-2xl font-bold mb-6 text-center text-gray-900">Generate QR Code</h2>
        <div className="flex flex-col gap-4 max-w-2xl mx-auto">
          <div>
            <label className="block text-sm font-medium text-gray-900 mb-1">Text to encode</label>
            <input
              type="text"
              value={text}
              onChange={(e) => setText(e.target.value)}
              placeholder="Enter text to encode"
              className="w-full p-3 bg-white border border-gray-300 rounded-lg text-gray-900 placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition duration-150 ease-in-out shadow-sm hover:shadow-md"
            />
          </div>
          <button
            onClick={() => setAdvancedOptionsVisible(!advancedOptionsVisible)}
            className="w-full p-3 bg-gray-50 border border-gray-300 rounded-lg text-gray-900 hover:bg-gray-200 transition-all text-sm font-medium"
          >
            {advancedOptionsVisible ? 'Hide advanced options ↑' : 'Show advanced options ↓'}
          </button>
          {advancedOptionsVisible && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div>
                <label className="block text-sm font-medium text-gray-900 mb-1">Version</label>
                <select
                  value={version}
                  onChange={(e) => setVersion(Number(e.target.value))}
                  className="w-full p-3 bg-gray-50 border border-gray-300 rounded-lg text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value={-1}>Auto</option>
                  {Array.from({ length: 10 }, (_, i) => i + 1).map((v) => (
                    <option key={v} value={v}>{v}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-900 mb-1">Error Correction</label>
                <select
                  value={errorCorrection}
                  onChange={(e) => setErrorCorrection(e.target.value)}
                  className="w-full p-3 bg-gray-50 border border-gray-300 rounded-lg text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="L">L</option>
                  <option value="M">M</option>
                  <option value="Q">Q</option>
                  <option value="H">H</option>
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-900 mb-1">Mask</label>
                <select
                  value={mask}
                  onChange={(e) => setMask(Number(e.target.value))}
                  className="w-full p-3 bg-gray-50 border border-gray-300 rounded-lg text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value={-1}>Auto</option>
                  {Array.from({ length: 8 }, (_, i) => i).map((m) => (
                    <option key={m} value={m}>{m}</option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-900 mb-1">Mode</label>
                <select
                  value={mode}
                  onChange={(e) => setMode(e.target.value)}
                  className="w-full p-3 bg-gray-50 border border-gray-300 rounded-lg text-gray-900 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                >
                  <option value="auto">Auto</option>
                  <option value="numeric">Numeric</option>
                  <option value="alphanumeric">Alphanumeric</option>
                  <option value="byte">Byte</option>
                  <option value="kanji">Kanji</option>
                </select>
              </div>
            </div>
          )}
          <PrimaryButton onClick={generateQR} disabled={isGenerating || !text}>
            {isGenerating ? 'Generating...' : 'Generate QR'}
          </PrimaryButton>
        </div>
      </div>

      {qrData && (
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <div className="bg-white rounded-xl p-6 shadow-xl h-fit">
            <h3 className="text-xl font-semibold mb-6 text-center text-gray-900">Generation Stages</h3>
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
            <div
              className="grid gap-0 bg-white p-4 rounded-lg mt-4 aspect-square"
              style={{ 
                gridTemplateColumns: `repeat(${qrData.size}, 1fr)`,
                gridTemplateRows: `repeat(${qrData.size}, 1fr)`
              }}
            >
              {currentMatrix?.modules.map((row, i) =>
                row.map((cell, j) => (
                  <div key={`${i}-${j}`} className="aspect-square">
                    <ModuleCell cell={!!cell} />
                  </div>
                ))
              )}
            </div>
            {currentStage > 0 && (() => {
              const previousMatrix = getMatrixForStage(currentStage - 1);
              if (previousMatrix && currentMatrix && JSON.stringify(previousMatrix.modules) === JSON.stringify(currentMatrix.modules)) {
                return <p className="mt-2 text-center text-sm text-gray-600">No modules have changed in this step</p>;
              }
              return null;
            })()}
          </div>
        </div>
      )}

      {qrData && (
        <CollapsibleSection title="Debug Logs">
          <div>
            <h4 className="font-bold mb-1">QR Generation Debug Logs</h4>
            <pre className="text-xs font-mono overflow-x-auto bg-gray-100 p-2 rounded">{qrData.debug_logs}</pre>
          </div>
        </CollapsibleSection>
      )}
    </div>
  );
} 