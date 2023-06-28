import React from "react";

const ExpenseTable = ({ data }) => {
  return (
    <div className="bg-white shadow-md rounded my-6">
      <table className="max-w-screen-lg w-full table-auto overflow-auto">
        <thead>
          <tr className="bg-gray-200 text-gray-600 uppercase text-sm leading-normal">
            <th className="py-3 px-6 text-left">Bank Account</th>
            <th className="py-3 px-6 text-left">Date</th>
            <th className="py-3 px-6 text-left">Narrative</th>
            <th className="py-3 px-6 text-left">Debit Amount</th>
            <th className="py-3 px-6 text-left">Credit Amount</th>
            <th className="py-3 px-6 text-left">Categories</th>
            <th className="py-3 px-6 text-left">Unnamed</th>
            <th className="py-3 px-6 text-left">Cats</th>
            <th className="py-3 px-6 text-left">Sum</th>
            <th className="py-3 px-6 text-left">Month</th>
          </tr>
        </thead>
        <tbody className="text-gray-600 text-sm font-light">
          {data.map((item, index) => (
            <tr
              key={index}
              className={(index % 2 === 0 ? "bg-white" : "bg-gray-50") + " border-b border-gray-200"}
            >
              <td className="py-3 px-6 text-left whitespace-nowrap">{item["Bank Account"]}</td>
              <td className="py-3 px-6 text-left">{item["Date"]}</td>
              <td className="py-3 px-6 text-left">{item["Narrative"]}</td>
              <td className="py-3 px-6 text-left">{item["Debit Amount"]}</td>
              <td className="py-3 px-6 text-left">{item["Credit Amount"]}</td>
              <td className="py-3 px-6 text-left">{item["Categories"]}</td>
              <td className="py-3 px-6 text-left">{item["Unnamed: 6"]}</td>
              <td className="py-3 px-6 text-left">{item["Cats"]}</td>
              <td className="py-3 px-6 text-left">{item["Sum"]}</td>
              <td className="py-3 px-6 text-left">{item["month"]}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default ExpenseTable;
