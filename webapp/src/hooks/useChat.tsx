import * as React from 'react';

// Types for the chat API
export interface ChatRequest {
  prompt: string;
  file: File;
}

export interface ChatResponse {
  response: string;
  has_masks: boolean;
  masked_image_path?: string;
  phrases?: string[];
}

export interface ChatError {
  error: string;
}

// Chat context type
interface ChatContextType {
  chat: (request: ChatRequest) => Promise<ChatResponse>;
  isLoading: boolean;
  error: string | null;
}

// Create the context
const ChatContext = React.createContext<ChatContextType | null>(null);

// Chat provider component
interface ChatProviderProps {
  children: React.ReactNode;
  apiUrl?: string;
}

export function ChatProvider({ children, apiUrl = 'http://localhost:9527' }: ChatProviderProps) {
  const [isLoading, setIsLoading] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  const chat = React.useCallback(async (request: ChatRequest): Promise<ChatResponse> => {
    setIsLoading(true);
    setError(null);

    try {
      // Use FormData for file upload
      const formData = new FormData();
      formData.append('file', request.file);
      formData.append('prompt', request.prompt);

      const response = await fetch(`${apiUrl}/chat`, {
        method: 'POST',
        // Don't set Content-Type header - browser will set it with boundary
        body: formData,
      });

      if (!response.ok) {
        const errorData: ChatError = await response.json().catch(() => ({
          error: `HTTP error! status: ${response.status}`,
        }));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const data: ChatResponse = await response.json();
      setIsLoading(false);
      return data;
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : 'An unknown error occurred';
      setError(errorMessage);
      setIsLoading(false);
      throw err;
    }
  }, [apiUrl]);

  return (
    <ChatContext.Provider value={{ chat, isLoading, error }}>
      {children}
    </ChatContext.Provider>
  );
}

// Hook to use the chat context
export function useChat(): ChatContextType {
  const context = React.useContext(ChatContext);

  if (!context) {
    throw new Error('useChat must be used within a ChatProvider');
  }

  return context;
}

