import { useState } from "react";
import IconButton from "./IconButton";
import { FaPlus, FaMinus } from "react-icons/fa";
import { motion, AnimatePresence } from "framer-motion";



interface CollapsibleSectionProps {
  title: string;
  children: React.ReactNode;
  initialExpanded?: boolean;
}

export default function CollapsibleSection({ title, children, initialExpanded = false }: CollapsibleSectionProps) {
  const [expanded, setExpanded] = useState(initialExpanded);
  return (
    <div className="bg-white rounded-xl p-6 shadow-xl mt-6">
      <div className="flex items-center justify-between">
        <h3 className="text-xl font-semibold mb-4 text-gray-900">{title}</h3>
        <IconButton onClick={() => setExpanded(!expanded)}>
          {expanded ? <FaMinus /> : <FaPlus />}
        </IconButton>
      </div>


      <AnimatePresence>
      {expanded && (
        <motion.div
          className="flex flex-col gap-4 overflow-hidden"
          initial={{ height: 0, opacity: 0 }}
          animate={{ height: 'auto', opacity: 1 }}
          exit={{ height: 0, opacity: 0 }}
          transition={{ duration: 0.3 }}
        >
          {children}
        </motion.div>
      )}
      </AnimatePresence>
    </div>
  );
} 