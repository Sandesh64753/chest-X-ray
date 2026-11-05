import { Layers, Eye, Brain, Cpu } from 'lucide-react';

const SECTIONS = [
  {
    icon: Brain,
    title: 'Grad-CAM — Gradient-Weighted Class Activation',
    content: `Gradient-weighted Class Activation Mapping computes backpropagated gradients of the predicted class score
    relative to the terminal feature map of DenseNet121. The resulting localization map reveals precisely which morphological
    features within the lung fields swayed the softmax layer toward a given pathology.`,
    items: [
      'Elevated activation zones indicate spatial regions influencing classification weights',
      'Attenuated activation zones represent background non-informative anatomical tissue',
      'Overlay synthesizes normalized heatmaps with 224×224 radiographic grayscale data',
      'Computed independently for the maximal argmax class output',
    ],
  },
  {
    icon: Eye,
    title: 'U-Net Anatomical Segmentation',
    content: `A deep U-Net architecture with symmetric contraction and expansion paths delineates the pulmonary boundary.
    Skip connections bridge low-level spatial detail directly to high-resolution feature synthesis, providing sharp boundary
    isolation even around retrocardiac and costophrenic recesses.`,
    items: [
      'Encoder–decoder architecture with multi-scale skip concatenation',
      'Normalized floating-point inputs [0, 1] rescaled to 224×224',
      'Sigmoid activation thresholded at 0.5 for binarized lung mask generation',
      'Parenchymal surface area percentage acts as a spatial coverage diagnostic',
    ],
  },
  {
    icon: Cpu,
    title: 'DenseNet121 Diagnostic Backbone',
    content: `DenseNet121 utilizes feed-forward dense connectivity where every layer receives concatenated feature maps
    from all preceding layers. This architectural paradigm mitigates vanishing gradients and strengthens feature reuse
    across deep representation stages.`,
    items: [
      'Pretrained on ImageNet with multi-stage fine-tuning on chest radiographs',
      'ImageNet mean and variance normalization applied per tensor channel',
      'Calibrated 4-head categorical softmax output',
      'Target taxonomy: COVID-19, Lung Opacity, Normal, and Viral Pneumonia',
    ],
  },
  {
    icon: Layers,
    title: 'Differential Metric Invariance',
    content: `When analyzing two longitudinal radiographs, our differential engine computes algebraic variance across
    softmax probability distributions, lung segmentation surface ratios, and spatial IoU of high-activation Grad-CAM zones.`,
    items: [
      'Logit delta: direct subtraction of categorical probabilities',
      'Surface variance: numeric delta in segmented lung volume ratio',
      'Spatial IoU: geometric intersection-over-union of core focus regions',
      'Metrics represent algorithmic output variance, not disease trajectory',
    ],
  },
];

export default function Explainability() {
  return (
    <div className="max-w-4xl space-y-6">
      <div className="border-b border-[#1f2127] pb-5">
        <h1 className="text-2xl font-bold text-white flex items-center gap-2.5">
          <Layers className="w-6 h-6 text-white" />
          Model Architecture & Explainability
        </h1>
        <p className="text-sm text-[#9ca3af] mt-1">
          Detailed technical review of neural network backbones, attention backpropagation, and segmentation methods.
        </p>
      </div>

      <div className="space-y-4">
        {SECTIONS.map(({ icon: Icon, title, content, items }) => (
          <div key={title} className="card p-6 space-y-4">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-lg bg-[#1a1c22] border border-[#2b2e38] flex items-center justify-center">
                <Icon className="w-5 h-5 text-white" />
              </div>
              <h2 className="text-base font-semibold text-white">{title}</h2>
            </div>
            <p className="text-sm text-[#d4d4d8] leading-relaxed">{content}</p>
            <ul className="space-y-2 pt-1 border-t border-[#1e2027]">
              {items.map((item) => (
                <li key={item} className="flex gap-2.5 text-xs text-[#a1a1aa] leading-relaxed">
                  <span className="text-white font-bold">›</span>
                  <span>{item}</span>
                </li>
              ))}
            </ul>
          </div>
        ))}
      </div>

      <div className="disclaimer-box">
        <div className="space-y-1">
          <span className="font-semibold text-white">Algorithmic Notice: </span>
          <span className="text-[#a1a1aa]">
            All heatmaps and masks illustrate mathematical feature weighting inside deep layers. They should not be
            confused with histological or pathology ground-truth examinations.
          </span>
        </div>
      </div>
    </div>
  );
}
