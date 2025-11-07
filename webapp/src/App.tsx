import { useState } from 'react';
import { ImageUpload } from './components/ImageUpload';
import { PromptSelector } from './components/PromptSelector';
import { ResultDisplay } from './components/ResultDisplay';
import { Button } from './components/ui/button';
import { Progress } from './components/ui/progress';
import { Satellite, ArrowLeft } from 'lucide-react';

type AppState = 'upload' | 'processing' | 'result';

export default function App() {
  const [state, setState] = useState<AppState>('upload');
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [prompt, setPrompt] = useState('');
  const [progress, setProgress] = useState(0);
  const [resultImage, setResultImage] = useState('');
  const [outputPrompt, setOutputPrompt] = useState('');

  const handleImageSelect = (file: File) => {
    setSelectedImage(file);
  };

  const handleClearImage = () => {
    setSelectedImage(null);
  };

  const handleSubmit = () => {
    if (!selectedImage || !prompt) return;

    setState('processing');
    setProgress(0);

    // Simulate processing with progress
    const interval = setInterval(() => {
      setProgress((prev) => {
        if (prev >= 100) {
          clearInterval(interval);
          // Simulate result
          const reader = new FileReader();
          reader.onloadend = () => {
            setResultImage(reader.result as string);
            setOutputPrompt(
              `Analysis completed successfully. The system has processed the satellite imagery and identified key areas based on your prompt: "${prompt}". Results are highlighted in the processed image.`
            );
            setState('result');
          };
          reader.readAsDataURL(selectedImage);
          return 100;
        }
        return prev + 10;
      });
    }, 300);
  };

  const handleDownload = () => {
    if (!resultImage) return;
    
    const link = document.createElement('a');
    link.href = resultImage;
    link.download = `georescue-analysis-${Date.now()}.png`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  const handleReset = () => {
    setState('upload');
    setSelectedImage(null);
    setPrompt('');
    setProgress(0);
    setResultImage('');
    setOutputPrompt('');
  };

  return (
    <div className="min-h-screen bg-zinc-950 text-white">
      <div className="max-w-2xl mx-auto px-4 py-6 pb-20">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          {state !== 'upload' && (
            <button
              onClick={handleReset}
              className="flex items-center gap-2 text-zinc-400 hover:text-zinc-200 transition-colors"
            >
              <ArrowLeft className="w-5 h-5" />
              <span>Back</span>
            </button>
          )}
          {state === 'upload' && <div />}
          <div className="flex items-center gap-2">
            <Satellite className="w-6 h-6 text-blue-500" />
            <h1 className="text-blue-500">GeoRescue</h1>
          </div>
        </div>

        {/* Subtitle */}
        {state === 'upload' && (
          <div className="mb-8 text-center">
            <h2 className="text-zinc-400 mb-2">
              AI-Powered Disaster Response Analysis
            </h2>
            <p className="text-zinc-500">
              Upload satellite imagery and get instant analysis for emergency response
            </p>
          </div>
        )}

        {/* Upload State */}
        {state === 'upload' && (
          <div className="space-y-6">
            <div>
              <h3 className="text-zinc-300 mb-4">1. Upload Satellite Image</h3>
              <ImageUpload
                onImageSelect={handleImageSelect}
                selectedImage={selectedImage}
                onClear={handleClearImage}
              />
            </div>

            {selectedImage && (
              <div>
                <h3 className="text-zinc-300 mb-4">2. Select Analysis Type</h3>
                <PromptSelector 
                  onPromptChange={setPrompt}
                  prompt={prompt}
                />
              </div>
            )}

            {selectedImage && prompt && (
              <Button
                onClick={handleSubmit}
                className="w-full bg-blue-600 hover:bg-blue-700 text-white py-6"
              >
                Start Analysis
              </Button>
            )}
          </div>
        )}

        {/* Processing State */}
        {state === 'processing' && (
          <div className="space-y-6 py-12">
            <div className="text-center">
              <div className="inline-flex p-4 bg-blue-600/20 rounded-full mb-4">
                <Satellite className="w-12 h-12 text-blue-500 animate-pulse" />
              </div>
              <h3 className="text-zinc-200 mb-2">
                Processing Satellite Imagery...
              </h3>
              <p className="text-zinc-400">
                Analyzing disaster response data
              </p>
            </div>
            
            <div className="space-y-2">
              <Progress value={progress} className="h-2" />
              <p className="text-center text-zinc-500">{progress}%</p>
            </div>

            <div className="space-y-2 text-center">
              <p className="text-zinc-400">
                {progress < 30 && 'Loading satellite data...'}
                {progress >= 30 && progress < 60 && 'Running AI analysis...'}
                {progress >= 60 && progress < 90 && 'Identifying key areas...'}
                {progress >= 90 && 'Finalizing results...'}
              </p>
            </div>
          </div>
        )}

        {/* Result State */}
        {state === 'result' && (
          <div>
            <h3 className="text-zinc-300 mb-4">Analysis Results</h3>
            <ResultDisplay
              imageUrl={resultImage}
              outputPrompt={outputPrompt}
              onDownload={handleDownload}
            />
            
            <Button
              onClick={handleReset}
              variant="outline"
              className="w-full mt-4 border-zinc-700 text-zinc-300 hover:bg-zinc-800"
            >
              Analyze Another Image
            </Button>
          </div>
        )}
      </div>
    </div>
  );
}
