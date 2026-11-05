import { AlertTriangle } from 'lucide-react';

export default function DisclaimerBox({ text }) {
  return (
    <div className="disclaimer-box flex items-start gap-3 bg-[#111216] border border-[#242730] rounded-lg p-4 text-xs text-[#d1d5db] leading-relaxed">
      <AlertTriangle className="w-4 h-4 text-[#d4af37] shrink-0 mt-0.5 opacity-90" />
      <div className="space-y-0.5">
        <span className="font-semibold text-white">Important Notice: </span>
        <span>{text}</span>
      </div>
    </div>
  );
}
