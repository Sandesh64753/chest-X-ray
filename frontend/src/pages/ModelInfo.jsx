import { useEffect, useState } from 'react';
import { Info, Cpu, Eye, Database, CheckCircle, XCircle } from 'lucide-react';
import { fetchModelInfo } from '../api/client';

export default function ModelInfo() {
  const [info, setInfo] = useState(null);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchModelInfo()
      .then(setInfo)
      .catch(() => setError('Unable to reach FastAPI backend inference service.'));
  }, []);

  if (error) {
    return (
      <div className="max-w-4xl">
        <div className="bg-[#1c1313] border border-[#3b1c1c] rounded-lg p-5 text-white text-sm">{error}</div>
      </div>
    );
  }

  if (!info) {
    return (
      <div className="max-w-4xl flex justify-center py-20">
        <div className="spinner" />
      </div>
    );
  }

  return (
    <div className="max-w-4xl space-y-6">
      <div className="border-b border-[#1f2127] pb-5">
        <h1 className="text-2xl font-bold text-white flex items-center gap-2.5">
          <Info className="w-6 h-6 text-white" />
          Engine Parameters & Weights
        </h1>
        <p className="text-sm text-[#9ca3af] mt-1">
          Runtime configuration, layer specifications, and training dataset metadata.
        </p>
      </div>

      {/* Engine Status */}
      <div className={`card px-6 py-4 flex items-center gap-3.5 ${
        info.models_loaded ? 'border-[#26382b]' : 'border-[#3a2020]'
      }`}>
        {info.models_loaded ? (
          <CheckCircle className="w-5 h-5 text-[#22c55e] shrink-0" />
        ) : (
          <XCircle className="w-5 h-5 text-[#ef4444] shrink-0" />
        )}
        <div>
          <p className="text-sm font-semibold text-white">
            {info.models_loaded ? 'Active Inference Engine (All Models Loaded)' : 'Engine Weights Unloaded'}
          </p>
          <p className="text-xs text-[#8e929f] mt-0.5">{info.app_name} Version {info.version} (TensorFlow & Keras 3.0 runtime)</p>
        </div>
      </div>

      {/* Classifier Spec */}
      <div className="card p-6 space-y-4">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-[#1a1c22] border border-[#2b2e38] flex items-center justify-center">
            <Cpu className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-base font-semibold text-white">Classification Architecture</h2>
        </div>

        <div>
          {[
            ['Network Backbone', info.models.classifier.architecture],
            ['Primary Task', info.models.classifier.task],
            ['Tensor Dimensions', info.models.classifier.input_size],
            ['Head Output Count', (info.models.classifier.classes?.length || 0).toString()],
            ['Training Regimen', info.models.classifier.training],
          ].map(([k, v]) => (
            <div key={k} className="stat-row">
              <span className="stat-label">{k}</span>
              <span className="stat-value text-right font-mono text-xs max-w-sm">{v}</span>
            </div>
          ))}
        </div>

        <div className="pt-2">
          <p className="text-xs text-[#8e929f] mb-2.5 font-medium">Target Taxonomy Classes</p>
          <div className="flex flex-wrap gap-2">
            {info.models.classifier.classes?.map((cls) => (
              <span
                key={cls}
                className="inline-flex items-center gap-2 px-3 py-1 rounded-md text-xs font-medium bg-[#1a1c22] border border-[#2c2f38] text-white"
              >
                <span className="w-2 h-2 rounded-full bg-white opacity-80" />
                {cls}
              </span>
            ))}
          </div>
        </div>
      </div>

      {/* Segmenter Spec */}
      <div className="card p-6 space-y-4">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-[#1a1c22] border border-[#2b2e38] flex items-center justify-center">
            <Eye className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-base font-semibold text-white">Segmentation Architecture</h2>
        </div>
        <div>
          {[
            ['Network Backbone', info.models.segmenter.architecture],
            ['Primary Task', info.models.segmenter.task],
            ['Tensor Dimensions', info.models.segmenter.input_size],
          ].map(([k, v]) => (
            <div key={k} className="stat-row">
              <span className="stat-label">{k}</span>
              <span className="stat-value font-mono text-xs">{v}</span>
            </div>
          ))}
        </div>
      </div>

      {/* Training Data */}
      <div className="card p-6 space-y-4">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-lg bg-[#1a1c22] border border-[#2b2e38] flex items-center justify-center">
            <Database className="w-5 h-5 text-white" />
          </div>
          <h2 className="text-base font-semibold text-white">Dataset & Preprocessing Protocol</h2>
        </div>
        <div className="text-xs md:text-sm text-[#d4d4d8] leading-relaxed space-y-3">
          <p>
            Trained on the <strong className="text-white">COVID-19 Radiography Dataset</strong> curated by Chowdhury et al.,
            incorporating posterior–anterior (PA) view radiography partitions with curated lung segmentation masks.
          </p>
          <p>
            Ingested image tensors are converted to 3-channel RGB float tensors at 224×224 resolution. Classification applies
            ImageNet channel-wise normalization; segmentation applies zero-to-one interval scaling.
          </p>
        </div>
      </div>

      <div className="disclaimer-box">{info.medical_disclaimer}</div>
    </div>
  );
}
