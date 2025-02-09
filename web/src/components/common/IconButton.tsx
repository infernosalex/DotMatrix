import React from 'react';

interface IconButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {}

const IconButton: React.FC<IconButtonProps> = ({ children, className = '', ...props }) => {
  const baseClasses = "p-2 bg-neutral-800 text-gray-200 rounded-lg hover:bg-neutral-700 disabled:opacity-50 disabled:cursor-not-allowed transition-colors duration-200 min-w-[40px] aspect-square flex items-center justify-center";
  return (
    <button className={`${baseClasses} ${className}`} {...props}>
      {children}
    </button>
  );
};

export default IconButton; 