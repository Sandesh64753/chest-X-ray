import { useState } from 'react';
import { GitCompare, RefreshCw, ArrowRight, TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { compareXrays } from '../api/client';
import ImageUploader from '../components/ImageUploader';
import ProbabilityBars from '../components/ProbabilityBars';
import DisclaimerBox from '../components/DisclaimerBox';
import Spinner from '../components/Spinner';

const CLASS_LABELS = {
  COVID: 'COVID-19',
  Lung_Opacity: 'Lung Opacity',
  Normal: 'Normal',
  'Viral Pneumonia': 'Viral Pneumonia',
};

function DeltaIcon({ d }) {
  if (d > 0.5) return <TrendingUp className="w-3.5 h-3.5 text-white opacity-90" />;
  if (d < -0.5) return <TrendingDown className="w-3.5 h-3.5 text-[#9ca3af]" />;
  return <Minus className="w-3.5 h-3.5 text-[#52525b]" />;
}

export default function Compare() {
  const [file1, setFile1] = useState(null);
  const [file2, setFile2] = useState(null);
  const [demo1, setDemo1] = useState(false);
  const [demo2, setDemo2] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(null);
  const [view1, setView1] = useState('gcam');
  const [view2, setView2] = useState('gcam');

  const getImage = (img, view) => {
    if (!img) return '';
    if (view === 'gcam') return img.gcam_overlay;
    if (view === 'seg') return img.seg_overlay;
    return img.raw;
  };

  const handleCompare = async () => {
    if ((!file1 && !demo1) || (!file2 && !demo2)) {
      setError('Please provide or select both radiographic images for comparison.');
      return;
    }
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      const res = await compareXrays({
        file1: file1 || undefined,
        file2: file2 || undefined,
        use_demo1: demo1,
        use_demo2: demo2,
      });
      setResult(res);
    } catch (e) {
      const msg = e?.response?.data?.detail || 'Comparison analysis failed. Please verify input files and retry.';
      setError(msg);
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFile1(null); setFile2(null);
    setDemo1(false); setDemo2(false);
    setResult(null); setError(null);
  };

  const CLASS_BADGE = {
    COVID: 'badge-red',
    Lung_Opacity: 'badge-amber',
    Normal: 'badge-green',
    'Viral Pneumonia': 'badge-blue',
  };

  const imgTabs = [
    { key: 'raw', label: 'Original' },
    { key: 'gcam', label: 'Grad-CAM' },
    { key: 'seg', label: 'Segmentation' },
  ];

  return (
    <div className="space-y-6">
      <div className="border-b border-[#1f2127] pb-5">
        <h1 className="text-2xl font-bold text-white flex items-center gap-2.5">
          <GitCompare className="w-6 h-6 text-white" />
          Comparative Differential
        </h1>
        <p className="text-sm text-[#9ca3af] mt-1">
          Perform differential assessment across two longitudinal chest radiographs.
        </p>
      </div>

      {/* Input Selection */}
      {!result && (
        <div className="card p-6 space-y-6">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <ImageUploader
              label="Baseline Image (Scan 1)"
              file={file1}
              onChange={(f) => { setFile1(f); setDemo1(false); }}
              demoLabel="Demo Baseline 1"
              onUseDemo={() => { setDemo1((v) => !v); setFile1(null); }}
              useDemo={demo1}
            />
            <ImageUploader
              label="Follow-up Image (Scan 2)"
              file={file2}
              onChange={(f) => { setFile2(f); setDemo2(false); }}
              demoLabel="Demo Follow-up 2"
              onUseDemo={() => { setDemo2((v) => !v); setFile2(null); }}
              useDemo={demo2}
            />
          </div>

          {error && (
            <div className="bg-[#1c1313] border border-[#3b1c1c] rounded-lg px-4 py-3 text-xs text-[#fee2e2]">
              {error}
            </div>
          )}

          <div className="flex gap-3">
            <button
              onClick={handleCompare}
              disabled={loading || ((!file1 && !demo1) || (!file2 && !demo2))}
              className="btn-primary"
            >
              {loading ? (
                <><div className="spinner w-3.5 h-3.5 !border-t-black" /> Computing Comparison...</>
              ) : (
                <><GitCompare className="w-4 h-4" /> Run Differential</>
              )}
            </button>
          </div>
        </div>
      )}

      {loading && <Spinner label="Performing dual inference & calculating differential matrix..." />}

      {/* Comparison Results */}
      {result && !loading && (
        <div className="space-y-6">
          {/* Comparison Status Bar */}
          <div className="card px-6 py-4 flex flex-wrap gap-4 items-center justify-between">
            <div className="flex items-center gap-3 text-sm">
              <span className={`badge ${CLASS_BADGE[result.image_1.predicted_class] ?? 'badge-slate'}`}>
                Scan 1: {result.image_1.predicted_class}
              </span>
              <ArrowRight className="w-4 h-4 text-[#71717a]" />
              <span className={`badge ${CLASS_BADGE[result.image_2.predicted_class] ?? 'badge-slate'}`}>
                Scan 2: {result.image_2.predicted_class}
              </span>
              <div className="h-4 w-[1px] bg-[#272932]" />
              {result.measurable_differences.predicted_class_changed ? (
                <span className="text-xs text-white font-medium bg-[#242630] border border-[#3b3f4e] px-2.5 py-1 rounded">
                  Classification Categorical Shift
                </span>
              ) : (
                <span className="text-xs text-[#a1a1aa] font-medium">Categorically Consistent</span>
              )}
            </div>
            <button onClick={handleReset} className="btn-outline text-xs py-1.5 px-3">
              <RefreshCw className="w-3.5 h-3.5" /> New Comparison
            </button>
          </div>

          {/* Side-by-Side Images */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {[1, 2].map((n) => {
              const imgData = n === 1 ? result.image_1 : result.image_2;
              const view = n === 1 ? view1 : view2;
              const setView = n === 1 ? setView1 : setView2;

              return (
                <div key={n} className="card p-5 space-y-3.5">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-semibold text-white">Scan {n}</span>
                    <span className={`badge ${CLASS_BADGE[imgData.predicted_class] ?? 'badge-slate'} text-xs`}>
                      {imgData.predicted_class} • {(imgData.confidence * 100).toFixed(1)}%
                    </span>
                  </div>

                  <div className="flex gap-1.5 text-xs">
                    {imgTabs.map(({ key, label }) => (
                      <button
                        key={key}
                        onClick={() => setView(key)}
                        className={`px-2.5 py-1 rounded font-medium transition-colors ${
                          view === key
                            ? 'bg-[#272932] text-white border border-[#3f4350]'
                            : 'bg-[#15161b] text-[#9ca3af] hover:text-white border border-[#21232b]'
                        }`}
                      >
                        {label}
                      </button>
                    ))}
                  </div>

                  <div className="img-panel">
                    <img
                      src={getImage(imgData.images, view)}
                      alt={`Scan ${n} ${view}`}
                      className="w-full"
                    />
                  </div>

                  <div className="text-xs text-[#a1a1aa] flex justify-between">
                    <span>Segmented lung ratio:</span>
                    <span className="font-mono text-white font-semibold">{imgData.lung_mask_area_pct.toFixed(2)}%</span>
                  </div>
                </div>
              );
            })}
          </div>

          {/* Delta Probability Matrix */}
          <div className="card p-6 space-y-4">
            <div className="section-title">Differential Probability Deltas (Scan 1 ➔ Scan 2)</div>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-[#22242a] text-[#8e929f]">
                    <th className="text-left pb-2.5 font-medium">Diagnostic Taxonomy</th>
                    <th className="text-right pb-2.5 font-medium">Baseline (Scan 1)</th>
                    <th className="text-right pb-2.5 font-medium">Follow-up (Scan 2)</th>
                    <th className="text-right pb-2.5 font-medium">Absolute Shift (Δ)</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-[#1c1e24]">
                  {result.delta_table.map(({ cls, p1, p2, d }) => (
                    <tr key={cls}>
                      <td className="py-2.5 font-medium text-white">{CLASS_LABELS[cls] ?? cls}</td>
                      <td className="py-2.5 text-right font-mono text-[#d4d4d8]">{p1.toFixed(1)}%</td>
                      <td className="py-2.5 text-right font-mono text-[#d4d4d8]">{p2.toFixed(1)}%</td>
                      <td className="py-2.5 text-right">
                        <span className="inline-flex items-center justify-end gap-1.5 font-mono font-semibold text-white">
                          <DeltaIcon d={d} />
                          <span>{d > 0 ? '+' : ''}{d.toFixed(1)}%</span>
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          {/* Probability distributions */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {[1, 2].map((n) => {
              const imgData = n === 1 ? result.image_1 : result.image_2;
              return (
                <div key={n} className="card p-6 space-y-4">
                  <div className="section-title">Scan {n} Logit Distribution</div>
                  <ProbabilityBars
                    probabilities={imgData.probabilities}
                    predictedClass={imgData.predicted_class}
                  />
                </div>
              );
            })}
          </div>

          {/* Measurable Diffs */}
          <div className="card p-6 space-y-4">
            <div className="section-title">Parametric Invariance & Variance</div>
            <div>
              <div className="stat-row">
                <span className="stat-label">Model Confidence Differential</span>
                <span className="stat-value font-mono">
                  {result.measurable_differences.confidence_delta > 0 ? '+' : ''}
                  {(result.measurable_differences.confidence_delta * 100).toFixed(2)}%
                </span>
              </div>
              <div className="stat-row">
                <span className="stat-label">Segmented Area Coverage Delta</span>
                <span className="stat-value font-mono">
                  {result.measurable_differences.lung_mask_area_pct_change > 0 ? '+' : ''}
                  {result.measurable_differences.lung_mask_area_pct_change.toFixed(2)}%
                </span>
              </div>
              <div className="stat-row">
                <span className="stat-label">Grad-CAM Spatial Alignment (IoU Overlap)</span>
                <span className="stat-value font-mono">
                  {result.measurable_differences.gradcam_region_overlap_iou != null
                    ? (result.measurable_differences.gradcam_region_overlap_iou * 100).toFixed(1) + '%'
                    : 'Disjoint Attention'}
                </span>
              </div>
            </div>
          </div>

          <DisclaimerBox text={result.disclaimer} />
        </div>
      )}
    </div>
  );
}
