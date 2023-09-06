import React from "react";

interface ColumnOrderInputProps {
  columns: {
    index: number; name: string; type: string 
}[];
  handleColumnsChange: (columns: { name: string; type: string }[]) => void;
}

const ColumnOrderInput: React.FC<ColumnOrderInputProps> = ({
  columns,
  handleColumnsChange,
}) => {
  const addColumn = () => {
    const newIndex = columns.length > 0 ? columns[columns.length - 1].index + 1 : 0;
    handleColumnsChange([...columns, { index: newIndex, name: "", type: "" }]);
  };

  const removeColumn = (index: number) => {
    const newColumns = [...columns];
    newColumns.splice(index, 1);
    handleColumnsChange(newColumns);
  };

  const handleInputChange = (
    event: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>,
    index: number,
    field: keyof { name: string; type: string }
  ) => {
    const newColumns = [...columns];
    newColumns[index][field] = event.target.value;
    handleColumnsChange(newColumns);
  };

  return (
    <section className="items-start">
      <h3 className="prose prose-invert">Tell us about how your columns are ordered</h3>
      <p className="prose prose-invert text-xs">(Description and category are mandatory to have. Order the columns in the way they appear left to right in your sheet)</p>
      <div className="w-full max-w-4xl bg-third p-6 rounded-xl shadow-lg prose-invert">
        <table className="table-auto">
          <thead>
            <tr>
              <th className="px-4 py-2"></th>
              <th className="px-4 py-2">Column Letter</th>
              <th className="px-4 py-2">Column Content</th>
              <th className="px-4 py-2"></th>
            </tr>
          </thead>
          <tbody>
        {columns.map((column, index) => (
          <tr key={`item-${column.index}`}>
            <td className="px-4 py-2">{column.index}</td>
                <td className="px-4 py-2">
                  <input
                    type="text"
                    value={column.name}
                    onChange={(event) => handleInputChange(event, index, "name")}
                    className="mt-1 text-gray-900 p-1 rounded-md"
                  />
                </td>
                <td className="px-4 py-2">
                  <select
                    value={column.type}
                    onChange={(event) => handleInputChange(event, index, "type")}
                    className="mt-1 text-gray-900 p-1 rounded-md"
                  >
                    <option value="">Select type</option>
                    <option value="source">Source</option>
                    <option value="date">Date</option>
                    <option value="description">Description</option>
                    <option value="amount">Amount</option>
                    <option value="category">Category</option>
                  </select>
                </td>
                <td className="px-4 py-2">
                  <button
                    className="prose-invert"
                    onClick={() => removeColumn(index)}
                  >
                    x
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <button
          className="text-lg font-bold leading-tight prose-invert p-2 m-2"
          onClick={addColumn}
        >
          +
        </button>
      </div>
    </section>
  );
};

export default ColumnOrderInput;
