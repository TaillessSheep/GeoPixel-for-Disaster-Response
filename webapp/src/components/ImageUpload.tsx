import { Upload, X } from 'lucide-react';
import { useState, useCallback } from 'react';
import { ImageWithFallback } from './figma/ImageWithFallback';

interface ImageUploadProps {
  onImageSelect: (file: File) => void;
  selectedImage: File | null;
  onClear: () => void;
}

export function ImageUpload({ onImageSelect, selectedImage, onClear }: ImageUploadProps) {
  const [isDragging, setIsDragging] = useState(false);
  const [preview, setPreview] = useState<string | null>(null);

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(true);
  }, []);

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    
    const files = e.dataTransfer.files;
    if (files && files[0] && files[0].type.startsWith('image/')) {
      onImageSelect(files[0]);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(files[0]);
    }
  }, [onImageSelect]);

  const handleFileInput = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files[0]) {
      onImageSelect(files[0]);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(files[0]);
    }
  }, [onImageSelect]);

  const handleClear = useCallback(() => {
    setPreview(null);
    onClear();
  }, [onClear]);

  if (selectedImage && preview) {
    return (
      <div className="relative w-full rounded-lg overflow-hidden border border-zinc-700">
        <ImageWithFallback 
          src={preview} 
          alt="Uploaded satellite imagery" 
          className="w-full h-64 object-cover"
        />
        <button
          onClick={handleClear}
          className="absolute top-2 right-2 bg-red-600 hover:bg-red-700 text-white rounded-full p-2 transition-colors"
          aria-label="Clear image"
        >
          <X className="w-4 h-4" />
        </button>
      </div>
    );
  }

  return (
    <div
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
      className={`border-2 border-dashed rounded-lg p-8 text-center transition-all ${
        isDragging 
          ? 'border-blue-500 bg-blue-500/10' 
          : 'border-zinc-700 bg-zinc-900/50 hover:border-zinc-600'
      }`}
    >
      <div className="flex flex-col items-center gap-4">
        <div className="p-4 bg-zinc-800 rounded-full">
          <Upload className="w-8 h-8 text-zinc-400" />
        </div>
        <div>
          <p className="text-zinc-300 mb-1">
            Drag and drop satellite image here
          </p>
          <p className="text-zinc-500">or</p>
        </div>
        <label className="cursor-pointer">
          <input
            type="file"
            accept="image/*"
            onChange={handleFileInput}
            className="hidden"
          />
          <span className="inline-block px-6 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors">
            Browse Files
          </span>
        </label>
      </div>
    </div>
  );
}
