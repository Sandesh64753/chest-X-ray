import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { ScanLine, GitCompare, Layers, Activity, AlertCircle, CheckCircle } from 'lucide-react';
import { fetchHealth } from '../api/client';

const CLASS_INFO = [
  { name: 'COVID-19', color: '#991b1b', desc: 'Bilateral ground-glass opacities, peripheral patchy distribution.' },
  { name: 'Lung Opacity', color: '#b45309', desc: 'Airspace consolidation from fluid, inflammatory exudate, or tissue.' },
  { name: 'Normal', color: '#166534', desc: 'Clear lung parenchyma, normal cardiopulmonary silhouette.' },
  { name: 'Viral Pneumonia', color: '#1d4ed8', desc: 'Diffuse interstitial infiltrates and peribronchial thickening.' },
];

const FEATURES = [
  {
    icon: ScanLine,
    title: 'Single X-ray Analysis',
    desc: 'Upload a radiographic image to generate multi-class predictions, segmented lung masks, and Grad-CAM attention.',
    to: '/analyze',
    cta: 'Open Analysis',
  },
  {
    icon: GitCompare,
    title: 'Comparative Differential',
    desc: 'Evaluate two sequential X-rays side by side with quantifiable probability and coverage delta metrics.',
    to: '/compare',
    cta: 'Open Comparison',
  },
  {
    icon: Layers,
    title: 'Explainability & Architecture',
    desc: 'Inspect DenseNet121 convolutional features, U-Net skip-connection masks, and gradient backpropagation maps.',
    to: '/explainability',
    cta: 'Review Details',
  },
];

export default function Dashboard() {
  const [health, setHealth] = useState(null);

  useEffect(() => {
    fetchHealth().then(setHealth).catch(() => null);
  }, []);

  return (
    <div className="space-y-10">
      {/* Hero Header */}
      <section className="pt-1">
        <div className="flex flex-col md:flex-row md:items-center justify-between gap-6 pb-6 border-b border-[#1f2127]">
          <div>
            <h1 className="text-3xl font-extrabold text-white tracking-tight">
              ChestVision Diagnostic System
            </h1>
            <p className="mt-2 text-[#9ca3af] text-sm md:text-base max-w-2xl leading-relaxed">
              Medical-grade machine learning platform for explainable chest radiography classification and lung delineation.
              Engineered with DenseNet121 and U-Net backbones.
            </p>
          </div>
          {health && (
            <div className={`flex items-center gap-2.5 px-3.5 py-1.5 rounded-full text-xs font-semibold border shrink-0 ${
              health.models_loaded
                ? 'bg-[#121814] border-[#203624] text-white'
                : 'bg-[#201414] border-[#3d2020] text-white'
            }`}>
              <div className={`w-2 h-2 rounded-full ${health.models_loaded ? 'bg-[#22c55e]' : 'bg-[#ef4444]'}`} />
              <span className="text-white">{health.models_loaded ? 'Inference Engine Ready' : 'Engine Offline'}</span>
            </div>
          )}
        </div>
      </section>

      {/* Model status warning if degraded */}
      {health && !health.models_loaded && (
        <div className="bg-[#1c1313] border border-[#3b1c1c] rounded-lg p-4 flex gap-3 items-start">
          <AlertCircle className="w-5 h-5 text-[#ef4444] shrink-0 mt-0.5" />
          <div>
            <p className="text-sm font-semibold text-white">Neural Network Weights Unavailable</p>
            <p className="text-xs text-[#9ca3af] mt-1">{health.error}</p>
          </div>
        </div>
      )}

      {/* Primary Workflows */}
      <section className="grid grid-cols-1 md:grid-cols-3 gap-5">
        {FEATURES.map(({ icon: Icon, title, desc, to, cta }) => (
          <div key={to} className="card p-6 flex flex-col justify-between hover:border-[#343844] transition-colors">
            <div className="space-y-4">
              <div className="w-10 h-10 rounded-lg bg-[#1a1c22] border border-[#2b2e38] flex items-center justify-center">
                <Icon className="w-5 h-5 text-white" />
              </div>
              <div>
                <h3 className="font-semibold text-white text-base tracking-tight">{title}</h3>
                <p className="text-xs md:text-sm text-[#9ca3af] mt-1.5 leading-relaxed">{desc}</p>
              </div>
            </div>
            <Link to={to} className="btn-outline mt-6 text-xs py-2 justify-center">
              {cta} →
            </Link>
          </div>
        ))}
      </section>

      {/* Diagnostic Categories */}
      <section className="space-y-4">
        <div>
          <h2 className="section-title">Diagnostic Classes</h2>
          <p className="section-subtitle">Trained taxonomy for clinical presentation</p>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
          {CLASS_INFO.map(({ name, color, desc }) => (
            <div key={name} className="card px-5 py-4 flex gap-3.5 items-start">
              <div
                className="w-2.5 h-2.5 rounded-full mt-1.5 shrink-0"
                style={{ backgroundColor: color }}
              />
              <div className="space-y-0.5">
                <p className="text-sm font-semibold text-white">{name}</p>
                <p className="text-xs text-[#9ca3af] leading-relaxed">{desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Inference Pipeline */}
      <section className="card p-6 space-y-5">
        <div className="flex items-center gap-2.5">
          <Activity className="w-4 h-4 text-white" />
          <h2 className="section-title">Execution Pipeline</h2>
        </div>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[
            { step: '01', label: 'Image Ingestion', desc: '224×224 bilinear rescaling & tensor casting' },
            { step: '02', label: 'DenseNet Inference', desc: 'Softmax probability across 4 pathology heads' },
            { step: '03', label: 'U-Net Masking', desc: 'Anatomical segmentation of lung parenchyma' },
            { step: '04', label: 'Grad-CAM Attention', desc: 'Gradient activation of terminal convolution layer' },
          ].map(({ step, label, desc }) => (
            <div key={step} className="p-3.5 rounded-lg bg-[#0e0f12] border border-[#1e2027]">
              <div className="text-[11px] font-mono text-white font-bold mb-1 opacity-80">{step}</div>
              <div className="text-xs font-semibold text-white">{label}</div>
              <div className="text-[11px] text-[#8e929f] mt-1 leading-relaxed">{desc}</div>
            </div>
          ))}
        </div>
      </section>

      {/* Disclaimer */}
      <section>
        <div className="disclaimer-box">
          <div className="space-y-1">
            <span className="font-semibold text-white">Notice of Intended Use: </span>
            <span className="text-[#a1a1aa]">
              ChestVision AI is an educational and algorithmic research prototype. Predictions, confidence ratings,
              segmentation boundaries, and heatmaps describe computational model behavior and must not replace professional
              radiological evaluation or clinical diagnosis.
            </span>
          </div>
        </div>
      </section>
    </div>
  );
}
