import { useState } from 'react';
import { ScanLine, Download, RefreshCw, ChevronDown, ChevronUp } from 'lucide-react';
import { analyzeXray, exportReport } from '../api/client';
import ImageUploader from '../components/ImageUploader';
import ProbabilityBars from '../components/ProbabilityBars';
import DisclaimerBox from '../components/DisclaimerBox';
import Spinner from '../components/Spinner';

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}

const CLASS_BADGE = {
  COVID: 'badge-red',
  Lung_Opacity: 'badge-amber',
  Normal: 'badge-green',
  'Viral Pneumonia': 'badge-blue',
};

export default function Analyze() {
  const [file, setFile] = useState(null);
  const [sample, setSample] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [activeTab, setActiveTab] = useState('gradcam');
  const [showAllImages, setShowAllImages] = useState(false);

  const handleAnalyze = async () => {
    if (!file && !sample) {
      setError('Please upload an X-ray scan or choose a demo sample.');
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await analyzeXray(file || undefined, sample || undefined);
      setResult(res);
      setActiveTab('gradcam');
    } catch (e) {
      const msg = e?.response?.data?.detail || 'Inference failed. Please verify the image file and retry.';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile(null);
    setSample(null);
    setResult(null);
    setError(null);
    setLoading(false);
  };

  const handleExport = async (fmt) => {
    try {
      const blob = await exportReport(fmt);
      downloadBlob(blob, `chestvision_report.${fmt}`);
    } catch {
      setError('No recent report in memory to export. Run an analysis first.');
    }
  };

  const predictedClass = result?.classification?.predicted_class;

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="border-b border-[#1f2127] pb-5">
        <h1 className="text-2xl font-bold text-white flex items-center gap-2.5">
          <ScanLine className="w-6 h-6 text-white" />
          Radiographic Analysis
        </h1>
        <p className="text-sm text-[#9ca3af] mt-1">
          Generate classification logits, lung segmentation boundaries, and Grad-CAM backpropagation heatmaps.
        </p>
      </div>

      {/* Input / Control Card */}
      {!result && (
        <div className="card p-6 space-y-5">
          <ImageUploader
            label="Input Radiograph (DICOM-derived or Standard Image)"
            file={file}
            onChange={(f) => { setFile(f); setSample(null); }}
            demoLabel="Demo Sample 1 (Pathology)"
            onUseDemo={() => { setSample(sample === 'sample_1' ? null : 'sample_1'); setFile(null); }}
            useDemo={sample === 'sample_1'}
          />

          {!file && sample !== 'sample_1' && (
            <button
              type="button"
              className="text-xs text-[#d4d4d8] hover:text-white underline font-medium block"
              onClick={() => { setSample('sample_2'); setFile(null); }}
            >
              Or load Demo Sample 2 →
            </button>
          )}

          {error && (
            <div className="bg-[#1c1313] border border-[#3b1c1c] rounded-lg px-4 py-3 text-xs text-[#fee2e2]">
              {error}
            </div>
          )}

          <div className="flex gap-3 pt-2">
            <button
              onClick={handleAnalyze}
              disabled={loading || (!file && !sample)}
              className="btn-primary"
            >
              {loading ? (
                <><div className="spinner w-3.5 h-3.5 !border-t-black" /> Computing...</>
              ) : (
                <><ScanLine className="w-4 h-4" /> Run Inference</>
              )}
            </button>
            {(file || sample) && (
              <button onClick={handleReset} className="btn-secondary">
                <RefreshCw className="w-4 h-4" />
                Reset
              </button>
            )}
          </div>
        </div>
      )}

      {/* Loading state */}
      {loading && <Spinner label="Executing forward pass (DenseNet121 + U-Net) & computing Grad-CAM gradients..." />}

      {/* Results View */}
      {result && !loading && (
        <div className="space-y-6">
          {/* Top Metric Bar */}
          <div className="card px-6 py-4 flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-3">
              <span className={`badge ${CLASS_BADGE[predictedClass] ?? 'badge-slate'} text-xs font-semibold px-3 py-1`}>
                {predictedClass}
              </span>
              <div className="h-4 w-[1px] bg-[#272932]" />
              <span className="text-xs text-[#9ca3af]">
                Confidence: <span className="font-mono font-bold text-white text-sm">{(result.classification.confidence * 100).toFixed(1)}%</span>
              </span>
              <div className="h-4 w-[1px] bg-[#272932]" />
              <span className="text-xs text-[#9ca3af]">
                Parenchyma Area: <span className="font-mono font-bold text-white text-sm">{result.segmentation.mask_area_pct.toFixed(1)}%</span>
              </span>
            </div>
            <div className="flex gap-2">
              <button onClick={() => handleExport('txt')} className="btn-secondary text-xs py-1.5 px-3">
                <Download className="w-3.5 h-3.5" /> TXT
              </button>
              <button onClick={() => handleExport('json')} className="btn-secondary text-xs py-1.5 px-3">
                <Download className="w-3.5 h-3.5" /> JSON
              </button>
              <button onClick={handleReset} className="btn-outline text-xs py-1.5 px-3">
                <RefreshCw className="w-3.5 h-3.5" /> New Analysis
              </button>
            </div>
          </div>

          {/* Grid Layout: Visuals vs Metrics */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Visuals Card */}
            <div className="card p-6 space-y-4">
              <div className="flex items-center justify-between">
                <div className="section-title">Visual Interpretations</div>
                <div className="flex gap-1.5 text-xs">
                  {['gradcam', 'segmentation', 'raw'].map((tab) => (
                    <button
                      key={tab}
                      onClick={() => setActiveTab(tab)}
                      className={`px-3 py-1.5 rounded font-medium transition-all ${
                        activeTab === tab
                          ? 'bg-[#272932] text-white border border-[#3f4350]'
                          : 'bg-[#15161b] text-[#9ca3af] hover:text-white border border-[#21232b]'
                      }`}
                    >
                      {tab === 'gradcam' ? 'Grad-CAM' : tab === 'segmentation' ? 'Lung Mask' : 'Original'}
                    </button>
                  ))}
                </div>
              </div>

              <div className="img-panel">
                {activeTab === 'gradcam' && (
                  <img src={result.images.gcam_overlay} alt="Grad-CAM overlay" className="w-full" />
                )}
                {activeTab === 'segmentation' && (
                  <img src={result.images.seg_overlay} alt="Segmentation overlay" className="w-full" />
                )}
                {activeTab === 'raw' && (
                  <img src={result.images.raw} alt="Original X-ray" className="w-full" />
                )}
              </div>

              <p className="text-xs text-[#a1a1aa] text-center font-mono">
                {activeTab === 'gradcam'
                  ? `Grad-CAM focus: ${result.gradcam.region_description}`
                  : activeTab === 'segmentation'
                  ? `U-Net delineated lung mask: ${result.segmentation.mask_area_pct.toFixed(2)}% of sensor field`
                  : 'Normalized input image'}
              </p>

              {/* Show Secondary Visuals */}
              <button
                className="text-xs text-[#8e929f] hover:text-white flex items-center gap-1.5 transition-colors pt-2"
                onClick={() => setShowAllImages((v) => !v)}
              >
                {showAllImages ? <ChevronUp className="w-3.5 h-3.5" /> : <ChevronDown className="w-3.5 h-3.5" />}
                {showAllImages ? 'Hide Raw Intermediates' : 'View Raw Heatmap & Binary Mask'}
              </button>

              {showAllImages && (
                <div className="grid grid-cols-2 gap-3 pt-2">
                  {[
                    { src: result.images.heatmap, label: 'Unblended Activation Map' },
                    { src: result.images.mask, label: 'Binarized Lung Mask' },
                  ].map(({ src, label }) => (
                    <div key={label} className="space-y-1">
                      <p className="text-[11px] text-[#9ca3af]">{label}</p>
                      <div className="img-panel">
                        <img src={src} alt={label} className="w-full" />
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>

            {/* Metrics Column */}
            <div className="space-y-6">
              <div className="card p-6 space-y-4">
                <div className="section-title">Softmax Probability Distribution</div>
                <ProbabilityBars
                  probabilities={result.classification.class_probabilities}
                  predictedClass={result.classification.predicted_class}
                />
              </div>

              <div className="card p-6 space-y-4">
                <div className="section-title">Quantitative Summary</div>
                <div>
                  <div className="stat-row">
                    <span className="stat-label">Model Classification</span>
                    <span className="stat-value">{result.classification.predicted_class}</span>
                  </div>
                  <div className="stat-row">
                    <span className="stat-label">Prediction Confidence</span>
                    <span className="stat-value font-mono">{(result.classification.confidence * 100).toFixed(2)}%</span>
                  </div>
                  <div className="stat-row">
                    <span className="stat-label">Parenchyma Area Ratio</span>
                    <span className="stat-value font-mono">{result.segmentation.mask_area_pct.toFixed(2)}%</span>
                  </div>
                  <div className="stat-row">
                    <span className="stat-label">Peak Activation Region</span>
                    <span className="stat-value text-right text-xs max-w-[200px] text-[#d4d4d8]">{result.gradcam.region_description}</span>
                  </div>
                  <div className="stat-row">
                    <span className="stat-label">Timestamp</span>
                    <span className="stat-value font-mono text-xs">{result.timestamp}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>

          <DisclaimerBox text={result.medical_disclaimer} />
        </div>
      )}
    </div>
  );
}
