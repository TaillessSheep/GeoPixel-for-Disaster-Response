import * as React from 'react';
import { Satellite } from 'lucide-react';
import { Progress } from './ui/progress';

interface ProcessingViewProps {
  progress: number;
}

export function ProcessingView({ progress }: ProcessingViewProps) {
  const getStatusMessage = () => {
    if (progress < 30) return 'Loading satellite data...';
    if (progress < 60) return 'Running AI analysis...';
    if (progress < 90) return 'Identifying key areas...';
    return 'Finalizing results...';
  };

  return (
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
          {getStatusMessage()}
        </p>
      </div>
    </div>
  );
}

