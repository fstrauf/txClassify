"use client";
import Link from "next/link";
import React, { useState } from "react";

export default function Demo() {
  // const [jsonKeyFile, setJsonKeyFile] = useState(null);
  const [spreadsheetLink, setSpreadsheetLink] = useState(
    "185s3wCfiHILwWIiWieKhpJYxs4l_VO8IX1IYX_QrFtw"
  );
  const [dataTabTraining, setDataTabTraining] = useState("A2:F200");
  const [dataTabClassify, setDataTabClassify] = useState("A1:C200");

  // const handleJsonKeyFileChange = (event: any) => {
  //   setJsonKeyFile(event.target.files[0]);
  // };

  const handleSpreadsheetLinkChange = (event: any) => {
    setSpreadsheetLink(event.target.value);
  };

  const handleDataTabTrainingChange = (event: any) => {
    setDataTabTraining(event.target.value);
  };

  const handleDataTabClassifyChange = (event: any) => {
    setDataTabClassify(event.target.value);
  };

  const handleTrainClick = async () => {
    // Provide the necessary parameters
    const spreadsheetId = spreadsheetLink;
    const range = dataTabTraining;

    const formData = new FormData();
    // if (jsonKeyFile) {
    //   formData.append("credentialsFile", jsonKeyFile);
    // }
    if (spreadsheetId) {
      formData.append("spreadsheetId", spreadsheetId);
    }
    if (range) {
      formData.append("range", range);
    }

    formData.append("customerName", "fs");

    try {
      const response = await fetch("/api/cleanAndPredict", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        console.log("Fetched data:", data);
      } else {
        console.error("API call failed with status:", response.status);
      }
    } catch (error) {
      console.error("An error occurred:", error);
    }
  };

  const handleClassifyClick = async () => {
    // Provide the necessary parameters
    const spreadsheetId = spreadsheetLink;
    const range = dataTabClassify;

    const formData = new FormData();
    // if (jsonKeyFile) {
    //   formData.append("credentialsFile", jsonKeyFile);
    // }
    if (spreadsheetId) {
      formData.append("spreadsheetId", spreadsheetId);
    }
    if (range) {
      formData.append("range", range);
    }

    formData.append("customerName", "fs");

    try {
      const response = await fetch("/api/cleanAndClassify", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        console.log("Fetched data:", data);
      } else {
        console.error("API call failed with status:", response.status);
      }
    } catch (error) {
      console.error("An error occurred:", error);
    }
  };

  return (
    <div className="flex flex-col min-h-screen bg-gradient-to-br from-first via-second to-third">
      <main className="flex-grow flex items-center justify-center p-10">
        <div className="w-full max-w-4xl bg-third p-6 rounded-xl shadow-lg text-white space-y-6">
          <h1 className="text-3xl font-bold leading-tight text-center">
            Step 1: Train Your Model
          </h1>
          <p>you need to authorise this email to edit your sheet (it's a service account that allows the script to read and write the classified expenses.</p>
          <p>The tool will currently only work with the <Link href='/fuck-you-money-sheet'>Fuck You Money Sheet</Link></p>
          <p>Add some classified expenses to the 'Details' tab prior to training</p>
          <label className="block">
            Spreadsheet ID:
            <input
              type="text"
              defaultValue="185s3wCfiHILwWIiWieKhpJYxs4l_VO8IX1IYX_QrFtw"
              value={spreadsheetLink}
              onChange={handleSpreadsheetLinkChange}
              className="mt-1 text-black w-full"
            />
          </label>

          <label className="block">
            Range:
            <input
              type="text"
              defaultValue="A2:G200"
              value={dataTabTraining}
              onChange={handleDataTabTrainingChange}
              className="mt-1 text-black w-full"
            />
          </label>

          <button
            onClick={handleTrainClick}
            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
          >
            Train
          </button>
        </div>
      </main>
      <main className="flex-grow flex items-center justify-center p-10">
        <div className="w-full max-w-4xl bg-third p-6 rounded-xl shadow-lg text-white space-y-6">
          <h1 className="text-3xl font-bold leading-tight text-center">
            Step 2: Classify your Expenses
          </h1>
          
          <p>you need to authorise this email to edit your sheet (it's a service account that allows the script to read and write the classified expenses.</p>
          <p>The tool will currently only work with the <Link href='/fuck-you-money-sheet'>Fuck You Money Sheet</Link></p>
          <p>Add your expenses to the sheet tab: 'new_dump' - use the format of the example values</p>
          <label className="block">
            Spreadsheet ID:
            <input
              type="text"
              defaultValue="185s3wCfiHILwWIiWieKhpJYxs4l_VO8IX1IYX_QrFtw"
              value={spreadsheetLink}
              onChange={handleSpreadsheetLinkChange}
              className="mt-1 text-black w-full"
            />
          </label>

          <label className="block">
            Range:
            <input
              type="text"
              defaultValue="C2:C200"
              value={dataTabClassify}
              onChange={handleDataTabClassifyChange}
              className="mt-1 text-black w-full"
            />
          </label>

          <button
            onClick={handleClassifyClick}
            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
          >
            Classify
          </button>
        </div>
      </main>
    </div>
  );
}
