import { useState, useEffect } from 'react';
import { ImageUpload } from './components/ImageUpload';
import { PromptSelector } from './components/PromptSelector';
import { ResultDisplay } from './components/ResultDisplay';
import { Header } from './components/Header';
import { ProcessingView } from './components/ProcessingView';
import { SectionTitle } from './components/SectionTitle';
import { WelcomeSection } from './components/WelcomeSection';
import { Button } from './components/ui/button';
import { useChat } from './hooks/useChat';

type AppState = 'upload' | 'processing' | 'result';

export default function App() {
  const { chat, isLoading, error } = useChat();
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

  const handleSubmit = async () => {
    if (!selectedImage || !prompt) return;

    setState('processing');
    setProgress(0);

    try {
      // Call the chat API with file directly (single request)
      setProgress(20);
      const response = await chat({
        prompt: prompt,
        file: selectedImage,
      });

      setProgress(80);
      
      // Handle the response
      setOutputPrompt(response.response);
      
      // If there's a masked image, load it
      if (response.has_masks && response.masked_image_path) {
        // Convert server path to URL
        // Server returns paths like "./vis_output/filename_masked.jpg"
        // We need to serve static files or convert to absolute URL
        const path = response.masked_image_path.startsWith('./') 
          ? response.masked_image_path.substring(2) 
          : response.masked_image_path;
        const imageUrl = `http://localhost:9527/${path}`;
        setResultImage(imageUrl);
      } else {
        // Fallback to original image if no masked version
        const reader = new FileReader();
        reader.onloadend = () => {
          setResultImage(reader.result as string);
        };
        reader.readAsDataURL(selectedImage);
      }
      
      setProgress(100);
      setState('result');
    } catch (err) {
      console.error('Error processing request:', err);
      setOutputPrompt(
        error || 'An error occurred while processing your request. Please try again.'
      );
      // Show original image on error
      const reader = new FileReader();
      reader.onloadend = () => {
        setResultImage(reader.result as string);
      };
      reader.readAsDataURL(selectedImage);
      setState('result');
    }
  };

  // Update progress based on loading state
  useEffect(() => {
    if (isLoading && state === 'processing') {
      // Simulate progress while loading
      const interval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 90) return prev; // Don't go to 100 until done
          return prev + 1;
        });
      }, 1100);
      return () => clearInterval(interval);
    }
  }, [isLoading, state]);

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
        <Header 
          showBackButton={state !== 'upload'} 
          onBackClick={handleReset}
        />

        {state === 'upload' && <WelcomeSection />}

        {/* Upload State */}
        {state === 'upload' && (
          <div className="space-y-6">
            <div>
              <SectionTitle>1. Upload Satellite Image</SectionTitle>
              <ImageUpload
                onImageSelect={handleImageSelect}
                selectedImage={selectedImage}
                onClear={handleClearImage}
              />
            </div>

            {selectedImage && (
              <div>
                <SectionTitle>2. Select Analysis Type</SectionTitle>
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
          <ProcessingView progress={progress} />
        )}

        {/* Error Display */}
        {error && state !== 'processing' && (
          <div className="mb-4 p-4 bg-red-900/20 border border-red-700 rounded-lg">
            <p className="text-red-400 text-sm">{error}</p>
          </div>
        )}

        {/* Result State */}
        {state === 'result' && (
          <div>
            <SectionTitle>Analysis Results</SectionTitle>
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
