import { useState } from 'react';
import { Textarea } from './ui/textarea';
import { Button } from './ui/button';
import { MapPin, Building, Waves, Navigation, AlertTriangle, Users } from 'lucide-react';

interface PromptSelectorProps {
  onPromptChange: (prompt: string) => void;
  prompt: string;
}

const COMMON_PROMPTS = [
  {
    icon: Building,
    label: 'Damaged Buildings',
    prompt: 'Identify and mark all damaged or collapsed buildings in the satellite imagery'
  },
  {
    icon: Waves,
    label: 'Flood Zones',
    prompt: 'Detect and highlight flooded areas and water accumulation zones'
  },
  {
    icon: Navigation,
    label: 'Safe Routes',
    prompt: 'Identify safe evacuation routes and accessible roads'
  },
  {
    icon: AlertTriangle,
    label: 'Hazard Areas',
    prompt: 'Mark potential hazard zones including debris, damaged infrastructure, and unsafe areas'
  },
  {
    icon: Users,
    label: 'Gathering Points',
    prompt: 'Locate potential safe gathering points and emergency shelter locations'
  },
  {
    icon: MapPin,
    label: 'Critical Infrastructure',
    prompt: 'Identify critical infrastructure including hospitals, power stations, and water facilities'
  }
];

export function PromptSelector({ onPromptChange, prompt }: PromptSelectorProps) {
  const [isCustom, setIsCustom] = useState(false);

  const handleQuickSelect = (selectedPrompt: string) => {
    onPromptChange(selectedPrompt);
    setIsCustom(false);
  };

  const handleCustomInput = (value: string) => {
    onPromptChange(value);
    setIsCustom(true);
  };

  return (
    <div className="space-y-4">
      <div>
        <label className="block text-zinc-300 mb-3">
          Select Analysis Type or Enter Custom Prompt
        </label>
        
        <div className="grid grid-cols-2 gap-2 mb-4">
          {COMMON_PROMPTS.map((item) => {
            const Icon = item.icon;
            const isSelected = prompt === item.prompt && !isCustom;
            return (
              <button
                key={item.label}
                onClick={() => handleQuickSelect(item.prompt)}
                className={`flex flex-col items-center gap-2 p-3 rounded-lg border transition-all ${
                  isSelected
                    ? 'bg-blue-600 border-blue-500 text-white'
                    : 'bg-zinc-800 border-zinc-700 text-zinc-300 hover:border-zinc-600 hover:bg-zinc-700'
                }`}
              >
                <Icon className="w-5 h-5" />
                <span className="text-xs text-center">{item.label}</span>
              </button>
            );
          })}
        </div>
      </div>

      <div>
        <label htmlFor="custom-prompt" className="block text-zinc-400 mb-2">
          Custom Analysis Prompt
        </label>
        <Textarea
          id="custom-prompt"
          value={isCustom ? prompt : ''}
          onChange={(e) => handleCustomInput(e.target.value)}
          placeholder="Describe what you want to analyze in the satellite imagery..."
          className="min-h-[100px] bg-zinc-900 border-zinc-700 text-zinc-100 placeholder:text-zinc-500 resize-none"
        />
      </div>

      {prompt && (
        <div className="p-3 bg-zinc-800/50 rounded-lg border border-zinc-700">
          <p className="text-xs text-zinc-400 mb-1">Selected Analysis:</p>
          <p className="text-zinc-200">{prompt}</p>
        </div>
      )}
    </div>
  );
}
