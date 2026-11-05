import { BookOpen, ExternalLink, Cpu, Database, Layers } from 'lucide-react';

const TECH_STACK = [
  { name: 'FastAPI Backend', desc: 'Asynchronous Python REST API framework with concurrency inference locks', icon: '⚡' },
  { name: 'TensorFlow & Keras', desc: 'Deep learning numerical execution runtime for graph evaluations', icon: '🧠' },
  { name: 'DenseNet121', desc: 'Densely Connected Convolutional Network for multi-class classification', icon: '🔬' },
  { name: 'U-Net Architecture', desc: 'Multi-scale encoder–decoder network for lung boundary segmentation', icon: '🫁' },
  { name: 'Grad-CAM Engine', desc: 'Mathematical gradient backpropagation for spatial attention maps', icon: '🎯' },
  { name: 'React & Vite', desc: 'Modern reactive component UI with rapid client-side routing', icon: '⚛️' },
  { name: 'Dual Black/Grey Theme', desc: 'Clean PACS-inspired medical workstation interface with white typography', icon: '🎨' },
  { name: 'Docker Orchestration', desc: 'Isolated multi-stage container deployment with health monitoring', icon: '🐳' },
];

const LIMITATIONS = [
  'Educational and algorithm demonstration model — strictly not for medical diagnosis.',
  'Trained on a specific open benchmark; patient-level covariate shift can affect performance.',
  'Grad-CAM heatmaps highlight model mathematical attention rather than clinical lesions.',
  'Binary pulmonary segmentation treats bilateral lung fields as a single contiguous region.',
  'Requires clinical radiologist review alongside full patient history for any medical decision.',
];

export default function About() {
  return (
    <div className="max-w-4xl space-y-6">
      <div className="border-b border-[#1f2127] pb-5">
        <h1 className="text-2xl font-bold text-white flex items-center gap-2.5">
          <BookOpen className="w-6 h-6 text-white" />
          System Overview & Technology Stack
        </h1>
        <p className="text-sm text-[#9ca3af] mt-1">
          Architectural paradigm, underlying frameworks, and scientific constraints.
        </p>
      </div>

      {/* Abstract */}
      <div className="card p-6 space-y-3.5">
        <h2 className="text-base font-semibold text-white flex items-center gap-2.5">
          <Cpu className="w-4 h-4 text-white" />
          Project Abstract
        </h2>
        <p className="text-xs md:text-sm text-[#d4d4d8] leading-relaxed">
          ChestVision AI is an explainable machine learning platform engineered to provide transparent visual and quantitative
          insights into deep neural networks processing chest radiographs. By pairing a DenseNet121 classifier with a U-Net
          segmenter and Grad-CAM interpretability, the system delivers holistic diagnostics without black-box opacity.
        </p>
        <p className="text-xs md:text-sm text-[#d4d4d8] leading-relaxed">
          The project utilizes a decoupled architecture: an asynchronous FastAPI Python microservice handles compute-intensive
          tensor inference, while a streamlined React client renders the PACS-styled clinical console in an ergonomic black and
          charcoal palette.
        </p>
      </div>

      {/* Stack Grid */}
      <div className="card p-6 space-y-4">
        <h2 className="text-base font-semibold text-white flex items-center gap-2.5">
          <Layers className="w-4 h-4 text-white" />
          Technical Stack
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-3.5">
          {TECH_STACK.map(({ name, desc, icon }) => (
            <div key={name} className="flex gap-3.5 p-3.5 rounded-lg bg-[#0e0f13] border border-[#1f2128]">
              <span className="text-lg leading-none mt-0.5">{icon}</span>
              <div>
                <p className="text-xs md:text-sm font-semibold text-white">{name}</p>
                <p className="text-xs text-[#8e929f] mt-0.5 leading-relaxed">{desc}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Dataset Card */}
      <div className="card p-6 space-y-3.5">
        <h2 className="text-base font-semibold text-white flex items-center gap-2.5">
          <Database className="w-4 h-4 text-white" />
          Dataset Provenance
        </h2>
        <p className="text-xs md:text-sm text-[#d4d4d8] leading-relaxed">
          Trained on the open-access <strong className="text-white">COVID-19 Radiography Database</strong> compiled by researchers
          from Qatar University and the University of Dhaka.
        </p>
        <div className="flex gap-2 flex-wrap pt-1">
          {['COVID-19', 'Lung Opacity', 'Normal', 'Viral Pneumonia'].map((cls) => (
            <span key={cls} className="badge badge-slate text-xs">{cls}</span>
          ))}
        </div>
      </div>

      {/* Known Limitations */}
      <div className="card p-6 space-y-3.5">
        <h2 className="text-base font-semibold text-white">Algorithmic Boundaries & Limitations</h2>
        <ul className="space-y-2">
          {LIMITATIONS.map((l) => (
            <li key={l} className="flex gap-2.5 text-xs text-[#d4d4d8] leading-relaxed">
              <span className="text-[#a1a1aa] font-bold">!</span>
              <span>{l}</span>
            </li>
          ))}
        </ul>
      </div>

      {/* Open Source */}
      <div className="card p-6 flex items-center justify-between">
        <div>
          <p className="text-sm font-semibold text-white">Open Source & Licensing</p>
          <p className="text-xs text-[#8e929f] mt-0.5">Educational MIT License</p>
        </div>
        <a
          href="https://github.com/Sandesh64753/chest-X-ray"
          target="_blank"
          rel="noopener noreferrer"
          className="btn-secondary text-xs"
        >
          <ExternalLink className="w-3.5 h-3.5" />
          Repository
        </a>
      </div>

      <div className="disclaimer-box">
        <div className="space-y-1">
          <span className="font-semibold text-white">Notice of Intended Use: </span>
          <span className="text-[#a1a1aa]">
            ChestVision AI is strictly an educational research tool. In all medical scenarios, diagnosis and clinical care
            must remain under the direct authority of licensed healthcare professionals.
          </span>
        </div>
      </div>
    </div>
  );
}
