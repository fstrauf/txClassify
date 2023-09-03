import { useState, ReactNode } from "react";

interface ConfigSectionProps {
  children: ReactNode;
}

export const ConfigSection = ({ children }: ConfigSectionProps) => {
  const [isConfigOpen, setIsConfigOpen] = useState(false);
  return (
    <>
      <button
        onClick={() => setIsConfigOpen(!isConfigOpen)}
        className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
      >
        {isConfigOpen ? "Hide" : "Show"} Configuration
      </button>

      {isConfigOpen && (
        <div className="mt-4 p-4 border rounded shadow">
          <h2 className="prose prose-invert text-lg mb-3 font-bold">
            Save your setting for future use!
          </h2>
          {children}
        </div>
      )}
    </>
  );
};
