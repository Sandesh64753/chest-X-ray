const CLASS_LABELS = {
  COVID: 'COVID-19',
  Lung_Opacity: 'Lung Opacity',
  Normal: 'Normal',
  'Viral Pneumonia': 'Viral Pneumonia',
};

// Subtle, non-neon, professional clinical palette
const MUTED_PALETTE = {
  COVID: '#991b1b',           // Deep Muted Crimson
  Lung_Opacity: '#b45309',    // Muted Amber/Bronze
  Normal: '#166534',          // Deep Forest Green
  'Viral Pneumonia': '#1d4ed8', // Deep Muted Steel Blue
};

export default function ProbabilityBars({ probabilities = {}, predictedClass = '' }) {
  const entries = Object.entries(probabilities).sort(([, a], [, b]) => b - a);

  return (
    <div className="space-y-4">
      {entries.map(([cls, prob]) => {
        const pct = (prob * 100).toFixed(1);
        const isTop = cls === predictedClass;
        const color = MUTED_PALETTE[cls] || '#52525b';

        return (
          <div key={cls} className="space-y-1.5">
            <div className="flex justify-between items-center text-sm">
              <span className={`flex items-center gap-2 ${isTop ? 'font-semibold text-white' : 'font-normal text-[#d4d4d8]'}`}>
                {CLASS_LABELS[cls] ?? cls}
                {isTop && (
                  <span className="text-[10px] font-bold px-1.5 py-0.5 rounded bg-[#242730] border border-[#3b3f4d] text-white tracking-wider">
                    PRIMARY
                  </span>
                )}
              </span>
              <span className={`font-mono text-sm ${isTop ? 'font-bold text-white' : 'text-[#a1a1aa]'}`}>
                {pct}%
              </span>
            </div>
            <div className="h-2 w-full bg-[#17181e] rounded-full overflow-hidden border border-[#23252d]">
              <div
                className="h-full rounded-full transition-all duration-500"
                style={{
                  width: `${pct}%`,
                  backgroundColor: isTop ? color : '#3f3f46',
                  opacity: isTop ? 1 : 0.65,
                }}
              />
            </div>
          </div>
        );
      })}
    </div>
  );
}
