import React, { useState } from "react";
import { Field, FieldArray } from 'formik'

const ExpenseTable = ({ data }) => {
  const [selectedCategories, setSelectedCategories] = useState({});

  // const handleCategoryChange = (event, index) => {
  //   const { value } = event.target;
  //   setSelectedCategories((prevState) => ({
  //     ...prevState,
  //     [index]: value,
  //   }));
  // };

  return (
    <div className="relative overflow-x-auto bg-white shadow-md rounded my-6 text-xs">
      <FieldArray
        name="expenses"
        render={(arrayHelpers) => (
          <>
            <table className="max-w-screen-2xl w-full table-auto overflow-auto">
              <thead>
                <tr className="bg-gray-200 text-gray-600 uppercase text-xs leading-normal">
                  <th className="py-3 px-6 text-left text-xs">Date</th>
                  <th className="py-3 px-6 text-left">Amount</th>
                  <th className="py-3 px-6 text-left">Narrative</th>
                  <th className="py-3 px-6 text-left">Categories</th>
                </tr>
              </thead>
              <tbody className="text-gray-600 text-xs font-light">
                {data.map((item, index) => (
                  <tr
                    key={index}
                    className={
                      (index % 2 === 0 ? "bg-white" : "bg-gray-50") +
                      " border-b border-gray-200"
                    }
                  >
                    <td className="py-2 px-3 text-left">{item?.Date}</td>
                    <td className="py-2 px-3 text-left">{item?.Amount}</td>
                    <td className="py-2 px-3 text-left">{item?.Narrative}</td>
                    <td className="py-2 px-3 text-left">
                    <Field
                        as="select"
                        name={`expenses.${index}.Categories`}
                      >
                        <option value={null}>none</option>
                        <option value="Groceries">Groceries</option>
                        <option value="Shopping">Shopping</option>
                        <option value="DinnerBars">Dinner/Bars</option>
                        <option value="Medical">Medical</option>
                        <option value="Transport">Transport</option>
                        <option value="Utility">Utility</option>
                        <option value="Travel">Travel</option>
                        <option value="Charlotte">Charlotte</option>
                        <option value="Business">Business</option>
                        <option value="Living">Living</option>
                      </Field>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </>
        )}
      />
    </div>
  );
};

export default ExpenseTable;
