'use client'
import React, { useState } from "react";

export default function Demo() {
  const [jsonKeyFile, setJsonKeyFile] = useState(null);
  const [spreadsheetLink, setSpreadsheetLink] = useState(
    "185s3wCfiHILwWIiWieKhpJYxs4l_VO8IX1IYX_QrFtw"
  );
  const [dataTab, setDataTab] = useState("A2:B200");

  const handleJsonKeyFileChange = (event: any) => {
    setJsonKeyFile(event.target.files[0]);
  };

  const handleSpreadsheetLinkChange = (event: any) => {
    setSpreadsheetLink(event.target.value);
  };

  const handleDataTabChange = (event:any) => {
    setDataTab(event.target.value);
  };

  async function callCleanAndPredict(training_data:[]) {
    const apiMode = 'saveTrainedData' //saveTrainedData || classify
    const customerName = 'Flo'
    const sheetApi = 'https://www.expensesorted.com/api/finishedTrainingHook'
    var body = {training_data, apiMode, customerName, sheetApi}
    try {
      console.log('Running Prediction')
      const response = await fetch("/api/cleanAndPredict", {
        method: "POST",
        body: JSON.stringify(body),
      });

      if (response.ok) {
        const data = await response.json();
        console.log("🚀 ~ file: page.tsx:37 ~ runPrediction ~ data:", data)

      } else {
        console.error("API call failed with status:", response.status);
        // Handle the error case
      }
    } catch (error) {
      console.error("An error occurred:", error);
      // Handle the error case
    }
  }

  
  // async function runPrediction(training_data: undefined) {
  //   const apiMode = 'saveTrainedData' //saveTrainedData || classify
  //   const customerName = 'Flo'
  //   const sheetApi = 'https://www.expensesorted.com/api/finishedTrainingHook'
  //   var body = {training_data, apiMode, customerName, sheetApi}
  //   try {
  //     console.log('Running Prediction')
  //     const response = await fetch("/api/runPrediction", {
  //       method: "POST",
  //       body: JSON.stringify(body),
  //     });

  //     if (response.ok) {
  //       const data = await response.json();
  //       console.log("🚀 ~ file: page.tsx:37 ~ runPrediction ~ data:", data)

  //     } else {
  //       console.error("API call failed with status:", response.status);
  //       // Handle the error case
  //     }
  //   } catch (error) {
  //     console.error("An error occurred:", error);
  //     // Handle the error case
  //   }
  // }

  const handleTrainClick = async () => {
    // Provide the necessary parameters
    const spreadsheetId = spreadsheetLink;
    const range = dataTab;

    const formData = new FormData();
    if (jsonKeyFile) {
      formData.append("credentialsFile", jsonKeyFile);
    }
    if (spreadsheetId) {
      formData.append("spreadsheetId", spreadsheetId);
    }
    if (range) {
      formData.append("range", range);
    }

    formData.append('customerName', 'fs')

    try {
      const response = await fetch("/api/cleanAndPredict", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        console.log("Fetched data:", data);
        // Continue with processing the fetched data as needed
        // await callCleanAndPredict(data)

      } else {
        console.error("API call failed with status:", response.status);
        // Handle the error case
      }
    } catch (error) {
      console.error("An error occurred:", error);
      // Handle the error case
    }
  };

  return (
    <div className="flex flex-col min-h-screen bg-gradient-to-br from-first via-second to-third">
      <main className="flex-grow flex items-center justify-center p-10">
        <div className="w-full max-w-4xl bg-third p-6 rounded-xl shadow-lg text-white space-y-6">
          <h1 className="text-3xl font-bold leading-tight text-center">
            Train Your Model
          </h1>

          <label className="block">
            Upload JSON Key File:
            <input
              type="file"
              onChange={handleJsonKeyFileChange}
              accept=".json"
              className="mt-1"
            />
          </label>

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
              value={dataTab}
              onChange={handleDataTabChange}
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
    </div>
  );
}


