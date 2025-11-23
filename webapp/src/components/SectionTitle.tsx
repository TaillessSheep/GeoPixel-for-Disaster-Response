import * as React from 'react';

interface SectionTitleProps {
  children: React.ReactNode;
  className?: string;
}

export function SectionTitle({ children, className = '' }: SectionTitleProps) {
  return (
    <h3 className={`text-zinc-300 mb-4 ${className}`}>
      {children}
    </h3>
  );
}

