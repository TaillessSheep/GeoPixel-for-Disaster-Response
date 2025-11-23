import * as React from 'react';
import { Satellite, ArrowLeft } from 'lucide-react';

interface HeaderProps {
  showBackButton?: boolean;
  onBackClick?: () => void;
}

export function Header({ showBackButton = false, onBackClick }: HeaderProps) {
  return (
    <div className="flex items-center justify-between mb-8">
      {showBackButton ? (
        <button
          onClick={onBackClick}
          className="flex items-center gap-2 text-zinc-400 hover:text-zinc-200 transition-colors"
        >
          <ArrowLeft className="w-5 h-5" />
          <span>Back</span>
        </button>
      ) : (
        <div />
      )}
      <div className="flex items-center gap-2">
        <Satellite className="w-6 h-6 text-blue-500" />
        <h1 className="text-blue-500">GeoRescue</h1>
      </div>
    </div>
  );
}

