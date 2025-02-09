import React, { memo } from 'react';
import { motion } from 'framer-motion';

interface ModuleCellProps {
  cell: boolean;
}

const ModuleCell: React.FC<ModuleCellProps> = memo(({ cell }) => (
  <motion.div
    animate={{ backgroundColor: cell ? '#000000' : '#FFFFFF', borderColor: cell ? '#111111' : '#EEEEEE' }}
    transition={{ duration: 0.2 }}
    className="w-full h-full border"
  />



));

export default ModuleCell; 