"use client";

import { useState } from "react";
import { Formik, Form } from "formik";
import ExpenseTable from "./ExpenseTable";

export default function UserInput() {
  const [initialValues, setInititalValues] = useState({
    filePath: "",
    expenses: [],
  });

  const handleUpdateClasses = (values) => {
    console.log(
      "🚀 ~ file: UserInput.js:19 ~ handleUpdateClasses ~ event:",
      values
    );
  };

  // const handleFileChange = (event, setFieldValue) => {
  //   const file = event.currentTarget.files[0];
  //   console.log("🚀 ~ file: UserInput.js:19 ~ handleFileChange ~ file:", file);
  //   setFieldValue("filePath", file.name);
  //   // setSelectedFileName(file.name);
  // };

  const submitData = async (values, { resetForm, setSubmitting }) => {
    setSubmitting(true);

    console.log(
      "🚀 ~ file: UserInput.js:34 ~ submitData ~ values.filePath:",
      values.filePath
    );
    if (values) {
      const formData = new FormData();
      formData.append("file", values.filePath);

      fetch("/api/classify", {
        method: "POST",
        body: formData,
      })
        .then((response) => response.json())
        .then((data) => {
          console.log("🚀 ~ file: UserInput.js:36 ~ .then ~ data:", data);
          setInititalValues({ filePath: values.filePath, expenses: data });
          // Handle the API response
          // setExpenseData(data);
          //reset formik state
        })
        .catch((error) => {
          // Handle any error that occurred during the API call
          console.error(error);
        });
    }
    setSubmitting(false);
  };

  return (
    <div className="flex flex-col">
      <Formik
        initialValues={initialValues}
        onSubmit={submitData}
        enableReinitialize
      >
        {({ isSubmitting, setFieldValue, values }) => (
          <Form>
            <div className="flex flex-col">
              <label htmlFor="fileUpload" className="block mb-4">
                Upload new transactions to be classified:
              </label>
              {/* <Field
                type="file"
                name='filePath'
                id="fileUpload"
                accept=".csv"
                className="mb-4"
                onChange={(event) => handleFileChange(event, setFieldValue)}
              /> */}
              <input
                id="file"
                name="file"
                type="file"
                onChange={(event) => {
                  setFieldValue("filePath", event.currentTarget.files[0]);
                }}
              />
              <div className="flex gap-4">
                <button
                  type="submit"
                  className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
                >
                  Classify
                </button>
                <button
                  type="button"
                  className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
                  onClick={() => handleUpdateClasses(values)}
                >
                  Update Categories
                </button>
              </div>
            </div>
            <ExpenseTable data={values?.expenses} />
          </Form>
        )}
      </Formik>
    </div>
  );
}
