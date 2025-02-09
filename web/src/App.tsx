import { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { QRGenerator } from './components/QRGenerator'
import { QRDecode } from './components/QRDecode'

function App() {
  const [currentTab, setCurrentTab] = useState("generator");

  return (
    <main className="min-h-screen min-w-full flex flex-col items-center bg-gradient-to-br from-gray-100 to-gray-200">
      <div className="container mx-auto px-4 py-12">
        <header className="mb-8 text-center">
          <h1 className="text-4xl font-extrabold text-gray-900">DotMatrix - Step by Step QR Code Generator & Decoder</h1>
        </header>
        <div className="flex justify-center space-x-4 mb-8">
          <button
            onClick={() => setCurrentTab("generator")}
            className={`px-4 py-2 border border-gray-300 font-semibold rounded-lg ${currentTab === "generator" ? "bg-blue-500 text-white" : "bg-gray-200 text-gray-800"}`}
          >
            QR Generator
          </button>
          <button
            onClick={() => setCurrentTab("decoder")}
            className={`px-4 py-2 border border-gray-300 font-semibold rounded-lg ${currentTab === "decoder" ? "bg-blue-500 text-white" : "bg-gray-200 text-gray-800"}`}
          >
            QR Decoder
          </button>
        </div>
        <AnimatePresence initial={false} mode="wait">
          {currentTab === "generator" && (
            <motion.section
              key="generator"
              initial={{ opacity: 0.5 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0.5 }}
              transition={{ duration: 0.15 }}
              className="mb-12"
            >
              <QRGenerator />
            </motion.section>
          )}
          {currentTab === "decoder" && (
            <motion.section
              key="decoder"
              initial={{ opacity: 0.5 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0.5 }}
              transition={{ duration: 0.15 }}
            >
              <QRDecode />
            </motion.section>
          )}
        </AnimatePresence>
      </div>
    </main>
  )
}

export default App
