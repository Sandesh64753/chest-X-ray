export default function Spinner({ label = 'Processing...', className = '' }) {
  return (
    <div className={`flex flex-col items-center justify-center gap-4 py-20 ${className}`}>
      <div className="w-10 h-10 border-2 border-[#262832] border-t-white rounded-full animate-spin" />
      <p className="text-sm text-white font-medium tracking-wide">{label}</p>
    </div>
  );
}
