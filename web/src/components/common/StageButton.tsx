import React from 'react';

interface StageButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  active: boolean;
  isLast?: boolean;
}

const StageButton: React.FC<StageButtonProps> = ({ active, isLast, children, ...props }) => {
  const baseClasses = "w-full text-left p-3 rounded-lg transition-all";
  const activeClasses = active ? "bg-blue-500 text-white font-medium ring-2 ring-blue-300" : "bg-neutral-800 text-gray-200 hover:bg-gray-700";
  const extraClasses = isLast ? "pl-2 border-l-10 border-blue-400" : "";
  
  return (
    <button className={`${baseClasses} ${activeClasses} ${extraClasses}`} {...props}>
      {children}
    </button>
  );
};

export default StageButton; 