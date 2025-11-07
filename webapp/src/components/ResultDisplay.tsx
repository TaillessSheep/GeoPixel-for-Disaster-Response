import { Download, CheckCircle } from 'lucide-react';
import { Button } from './ui/button';
import { ImageWithFallback } from './figma/ImageWithFallback';

interface ResultDisplayProps {
  imageUrl: string;
  outputPrompt?: string;
  onDownload: () => void;
}

export function ResultDisplay({ imageUrl, outputPrompt, onDownload }: ResultDisplayProps) {
  return (
    <div className="space-y-4">
      <div className="flex items-center gap-2 text-green-500 mb-4">
        <CheckCircle className="w-5 h-5" />
        <span>Analysis Complete</span>
      </div>

      <div className="relative rounded-lg overflow-hidden border border-zinc-700">
        <ImageWithFallback 
          src={imageUrl} 
          alt="Processed result" 
          className="w-full h-auto"
        />
      </div>

      {outputPrompt && (
        <div className="p-4 bg-zinc-800/50 rounded-lg border border-zinc-700">
          <p className="text-xs text-zinc-400 mb-2">Analysis Summary:</p>
          <p className="text-zinc-200">{outputPrompt}</p>
        </div>
      )}

      <Button 
        onClick={onDownload}
        className="w-full bg-blue-600 hover:bg-blue-700 text-white"
      >
        <Download className="w-4 h-4 mr-2" />
        Download Result
      </Button>
    </div>
  );
}
