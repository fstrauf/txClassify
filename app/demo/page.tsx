"use client";
import React, { useState } from "react";
import Instructions from "./instructions";
import SpreadSheetInput from "./spreadSheetInput";

export default function Demo() {
  const [spreadsheetLink, setSpreadsheetLink] = useState(
    "185s3wCfiHILwWIiWieKhpJYxs4l_VO8IX1IYX_QrFtw"
  );
  const [dataTabTraining, setDataTabTraining] = useState("A2:F200");
  const [dataTabClassify, setDataTabClassify] = useState("A1:C200");

  const [statusText, setStatusText] = useState("");

  const handleSpreadsheetLinkChange = (event: any) => {
    setSpreadsheetLink(event.target.value);
  };

  const handleTrainClick = async () => {
    const spreadsheetId = spreadsheetLink;
    const range = dataTabTraining;
    const formData = new FormData();

    if (spreadsheetId) {
      formData.append("spreadsheetId", spreadsheetId);
    }
    if (range) {
      formData.append("range", range);
    }

    // formData.append("customerName", spreadsheetId);

    try {
      setStatusText(`Training started based on sheet ${spreadsheetId}`)
      const response = await fetch("/api/cleanAndPredict", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        console.log("Fetched data:", data);
        setStatusText(`Training still running - this can take a couple of minutes`)
      } else {
        console.error("API call failed with status:", response.status);
      }
    } catch (error) {
      console.error("An error occurred:", error);
    }
  };

  const handleClassifyClick = async () => {
    const spreadsheetId = spreadsheetLink;
    const range = dataTabClassify;

    const formData = new FormData();

    if (spreadsheetId) {
      // formData.append("spreadsheetId", spreadsheetId);
    }
    if (range) {
      formData.append("range", range);
    }

    formData.append("spreadsheetId", spreadsheetId);

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
    <main className="flex flex-col min-h-screen bg-gradient-to-br from-first via-second to-third">
      <div className="flex-grow flex items-center justify-center p-10">
        <div className="w-full max-w-4xl bg-third p-6 rounded-xl shadow-lg text-white space-y-6">
          <h1 className="text-3xl font-bold leading-tight text-center">
            Step 1: Train Your Model
          </h1>
          <Instructions />
          <SpreadSheetInput
            spreadsheetLink={spreadsheetLink}
            handleSpreadsheetLinkChange={handleSpreadsheetLinkChange}
          />
          <p className="prose prose-invert">
            Range A2:F200 of sheet 'Expense Detail' is selected
          </p>
          {/* <label className="block">
            Spreadsheet ID:
            <input
              type="text"
              defaultValue="185s3wCfiHILwWIiWieKhpJYxs4l_VO8IX1IYX_QrFtw"
              value={spreadsheetLink}
              onChange={handleSpreadsheetLinkChange}
              className="mt-1 text-black w-full"
            />
          </label> */}

          {/* <label className="block">
            Range:
            <input
              type="text"
              defaultValue="A2:G200"
              value={dataTabTraining}
              onChange={handleDataTabTrainingChange}
              className="mt-1 text-black w-full"
            />
          </label> */}
          <div className="flex gap-3">
            <button
              onClick={handleTrainClick}
              className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
            >
              Train
            </button>
            {/* <p className="prose prose-invert text-xs">{statusText}</p> */}
          </div>
        </div>
      </div>
      <div className="flex-grow flex items-center justify-center p-10">
        <div className="w-full max-w-4xl bg-third p-6 rounded-xl shadow-lg text-white space-y-6">
          <h1 className="text-3xl font-bold leading-tight text-center">
            Step 2: Classify your Expenses
          </h1>
          <Instructions />
          <SpreadSheetInput
            spreadsheetLink={spreadsheetLink}
            handleSpreadsheetLinkChange={handleSpreadsheetLinkChange}
          />
          <p className="prose prose-invert">
            Range A1:C200 of sheet 'new_dump' is selected
          </p>
          {/* <label className="block">
            Spreadsheet ID:
            <div className="m-6">
              <Image
                width={1620 / 2.5}
                height={82 / 2.5}
                src="/f-you-sheet-id.png"
                className="rounded-md"
                alt="Add your income to the sheet"
              />
            </div>
            <input
              type="text"
              defaultValue="185s3wCfiHILwWIiWieKhpJYxs4l_VO8IX1IYX_QrFtw"
              value={spreadsheetLink}
              onChange={handleSpreadsheetLinkChange}
              className="mt-1 text-black w-full"
            />
          </label> */}

          {/* <label className="block">
              Range:
              <input
                type="text"
                defaultValue="C2:C200"
                value={dataTabClassify}
                onChange={handleDataTabClassifyChange}
                className="mt-1 text-black w-full"
              />
            </label> */}

          <button
            onClick={handleClassifyClick}
            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
          >
            Classify
          </button>
        </div>
      </div>
    </main>
  );
}
