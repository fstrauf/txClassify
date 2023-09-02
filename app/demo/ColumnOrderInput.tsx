import React from "react";

interface ColumnOrderInputProps {
  columns: { name: string; type: string }[];
  handleColumnsChange: (columns: { name: string; type: string }[]) => void;
}

const ColumnOrderInput: React.FC<ColumnOrderInputProps> = ({
  columns,
  handleColumnsChange,
}) => {
  const addColumn = () => {
    handleColumnsChange([...columns, { name: "", type: "" }]);
  };

  const removeColumn = (index: number) => {
    const newColumns = [...columns];
    newColumns.splice(index, 1);
    handleColumnsChange(newColumns);
  };

  const handleInputChange = (
    event: React.ChangeEvent<HTMLInputElement | HTMLSelectElement>,
    index: number,
    field: keyof (typeof columns)[index]
  ) => {
    const newColumns = [...columns];
    newColumns[index][field] = event.target.value;
    handleColumnsChange(newColumns);
  };

  return (
    <section className="items-start">
      <h2>How do the columns in your sheet look like?</h2>

      <div className="w-full max-w-4xl bg-third p-6 rounded-xl shadow-lg text-black">
        {columns.map((column, index) => (
          <div
            className="leading-tight p-2 items-center flex"
            key={`item-${index}`}
          >
            <div className="flex gap-4 items-center">
              <p className="text-white text-center">{index + 1}</p>
              <input
                type="text"
                value={column.name}
                onChange={(event) => handleInputChange(event, index, "name")}
                className="mt-1 text-black p-1 rounded-md"
              />
              <select
                value={column.type}
                onChange={(event) => handleInputChange(event, index, "type")}
                className="mt-1 text-black p-1 rounded-md"
              >
                <option value="">Select type</option>
                <option value="source">Source</option>
                <option value="date">Date</option>
                <option value="narrative">Description</option>
                <option value="amount">Amount</option>
                {/* <option value="credit">Credit</option> */}
                <option value="categories">Categories</option>
              </select>
              <button
                className="text-white"
                onClick={() => removeColumn(index)}
              >
                x
              </button>
            </div>
          </div>
        ))}
        <button
          className="text-lg font-bold leading-tight text-white p-2 m-2"
          onClick={addColumn}
        >
          +
        </button>
      </div>
    </section>
  );
};

export default ColumnOrderInput;
