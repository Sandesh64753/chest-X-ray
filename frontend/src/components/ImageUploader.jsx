import { useCallback, useState, useRef } from 'react';
import { Upload, X, ImageIcon } from 'lucide-react';

export default function ImageUploader({
  label,
  file,
  onChange,
  demoLabel,
  onUseDemo,
  useDemo,
  disabled,
}) {
  const [dragging, setDragging] = useState(false);
  const inputRef = useRef(null);

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setDragging(false);
      const dropped = e.dataTransfer.files[0];
      if (dropped) onChange(dropped);
    },
    [onChange]
  );

  const preview = file ? URL.createObjectURL(file) : null;

  return (
    <div className="space-y-2.5">
      <label className="block text-sm font-semibold text-white tracking-wide">{label}</label>

      {preview || useDemo ? (
        <div className="relative">
          <div className="img-panel h-52 flex items-center justify-center bg-[#070709] border border-[#262832]">
            {preview ? (
              <img src={preview} alt="Preview" className="max-h-52 object-contain" />
            ) : (
              <div className="text-center text-[#9ca3af] py-8">
                <ImageIcon className="w-10 h-10 mx-auto mb-2 opacity-60 text-white" />
                <p className="text-sm font-medium text-white">Demo sample selected</p>
                <p className="text-xs text-[#8e929f] mt-0.5">Ready for inference</p>
              </div>
            )}
          </div>
          <button
            type="button"
            onClick={() => { onChange(null); if (onUseDemo && useDemo) onUseDemo(); }}
            className="absolute top-2.5 right-2.5 w-7 h-7 rounded-full bg-[#181a20]/90 hover:bg-[#252832] text-white border border-[#3b3e4a] flex items-center justify-center transition-colors"
            title="Remove selection"
          >
            <X className="w-3.5 h-3.5" />
          </button>
          {preview && (
            <p className="text-xs text-[#9ca3af] mt-1.5 truncate font-mono">{file?.name}</p>
          )}
        </div>
      ) : (
        <div
          className={`dropzone ${dragging ? 'active' : ''} ${disabled ? 'opacity-40 cursor-not-allowed' : ''}`}
          onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
          onDragLeave={() => setDragging(false)}
          onDrop={handleDrop}
          onClick={() => !disabled && inputRef.current?.click()}
        >
          <div className="w-10 h-10 rounded-full bg-[#1a1c22] border border-[#2b2e38] flex items-center justify-center mx-auto mb-3">
            <Upload className="w-5 h-5 text-white" />
          </div>
          <p className="text-sm font-semibold text-white">Drop X-ray scan here or click to browse</p>
          <p className="text-xs text-[#8e929f] mt-1">Accepts PNG, JPG, JPEG, BMP, WebP (up to 15 MB)</p>
          <input
            ref={inputRef}
            type="file"
            className="hidden"
            accept=".png,.jpg,.jpeg,.bmp,.webp"
            disabled={disabled}
            onChange={(e) => {
              const f = e.target.files?.[0];
              if (f) onChange(f);
            }}
          />
        </div>
      )}

      {demoLabel && onUseDemo && !file && !useDemo && (
        <button
          type="button"
          onClick={onUseDemo}
          disabled={disabled}
          className="btn-secondary w-full text-xs py-2.5 justify-center mt-1"
        >
          <ImageIcon className="w-3.5 h-3.5" />
          Load {demoLabel}
        </button>
      )}
    </div>
  );
}
