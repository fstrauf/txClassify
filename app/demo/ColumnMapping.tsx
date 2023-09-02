import React, { useState, ChangeEvent } from 'react';

interface ColumnMappingProps {
  googleSheetColumns: string[];
  expectedColumns: string[];
}

const ColumnMapping = ({ googleSheetColumns, expectedColumns }: ColumnMappingProps) => {
  const [mapping, setMapping] = useState<Record<string, string>>({});

  const handleMappingChange = (googleSheetColumn: string, event: ChangeEvent<HTMLSelectElement>) => {
    setMapping({
      ...mapping,
      [googleSheetColumn]: event.target.value,
    });
  };

  return (

    <div>
      <h2>Tell us about the columns (skip if using our template sheet)</h2>
      {googleSheetColumns.map((column: string) => (
        <div key={column}>
          <label>
            {column}
            <select onChange={(e) => handleMappingChange(column, e)}>
              <option value="">Select...</option>
              {expectedColumns.map((expectedColumn: string) => (
                <option key={expectedColumn} value={expectedColumn}>
                  {expectedColumn}
                </option>
              ))}
            </select>
          </label>
        </div>
      ))}
      <button onClick={() => console.log(mapping)}>Save Mapping</button>
    </div>
  );
};

export default ColumnMapping;