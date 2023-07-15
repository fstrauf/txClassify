"use client";

import { useState } from "react";
import { Formik, Form } from "formik";
import ExpenseTable from "./ExpenseTable";
// import { createServerComponentClient } from "@supabase/auth-helpers-nextjs";
// import { cookies } from "next/headers";



export default async function UserInput() {
  const [initialValues, setInititalValues] = useState({
    filePath: "",
    expenses: [],
  });

  // const supabase = createServerComponentClient({ cookies });

  // const { data: expenses } = await supabase.from("expenses").select();


  // var faunadb = require("faunadb");
  // var q = faunadb.query;

  // var client = new faunadb.Client({
  //   secret: "fnAFIypPOgAARE8lGiZDiwplIFu6-BIEj2-NADbN",
  //   // NOTE: Use the correct endpoint for your database's Region Group.
  //   endpoint: "https://db.fauna.com:443/",
  // });

  // var createP = client.query(
  //   q.Create(q.Collection("Expenses"), {
  //     data: {
  //       date: "2020-01-02",
  //       description: "new",
  //       debitAmount: 150,
  //       creditAmount: 0,
  //     },
  //   })
  // );

  const handleUpdateClasses = (values) => {
    if (values) {
      const requestBody = values.expenses;

      fetch("/api/retrain", {
        method: "POST",
        body: JSON.stringify(requestBody),
        headers: {
          "Content-Type": "application/json",
        },
      })
        .then((response) => response.json())
        .then((data) => {
          console.log("🚀 ~ file: UserInput.js:36 ~ .then ~ data:", data);
          console.log("Successfully retrained based on user inputs");
        })
        .catch((error) => {
          console.error(error);
        });
    }
  };

  const handleExportData = (values) => {
    if (values) {
      const requestBody = values.expenses;

      fetch("/api/convertToCSV", {
        method: "POST",
        body: JSON.stringify(requestBody),
        headers: {
          "Content-Type": "application/json",
        },
      })
        .then((response) => response.text())
        .then((csvData) => {
          const csvFile = new Blob([csvData], { type: "text/csv" });
          const csvURL = URL.createObjectURL(csvFile);
          const downloadLink = document.createElement("a");
          downloadLink.href = csvURL;
          downloadLink.download = "exported_data.csv";
          downloadLink.click();
          URL.revokeObjectURL(csvURL);
          console.log("CSV file exported successfully");
        })
        .catch((error) => {
          console.error(error);
        });
    }
  };

  const adminUpload = (values) => {
    if (values) {
      const formData = new FormData();
      formData.append("file", values.filePath);
      fetch("/api/adminUpload", {
        method: "POST",
        body: formData,
      })
        .then((response) => response.json())
        .then((data) => {
          console.log("🚀 ~ file: UserInput.js:93 ~ .then ~ data:", data);
          // setInititalValues({ filePath: values.filePath, expenses: data });
        })
        .catch((error) => {
          console.error(error);
        });
    }
  };

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
        })
        .catch((error) => {
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
                <button
                  type="button"
                  className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
                  onClick={() => handleExportData(values)}
                >
                  Export New Data
                </button>
              </div>
            </div>
            <ExpenseTable data={values?.expenses} />
            <h2>Admin Stuff</h2>
            <button
              type="button"
              className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
              onClick={() => adminUpload(values)}
            >
              Upload all past classified expenses
            </button>
          </Form>
        )}
      </Formik>

    </div>
  );
}
