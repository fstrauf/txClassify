"use client";
import React, { useEffect, useState } from "react";
import Instructions from "./instructions";
import SpreadSheetInput from "./spreadSheetInput";
import RangeInput from "./rangeInput";
import { createClient } from "@supabase/supabase-js";
import { useUser } from "@auth0/nextjs-auth0/client";
import ProtectedPage from "../../components/ProtectedPage";
import { SaveConfigButton } from "../../components/buttons/save-config-button";

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY
);

const Demo = () => {
  const { user } = useUser();

  const [statusText, setStatusText] = useState("");
  const [config, setConfig] = useState({
    // expenseSheetId: "185s3wCfiHILwWIiWieKhpJYxs4l_VO8IX1IYX_QrFtw",
    // trainingRange: "Expense-Detail!A2:F200",
    // categorisationRange: "new_dump!A1:C200",
  });

  const [data, setData] = useState({});

  useEffect(() => {
    const fetchData = async () => {
      console.log("🚀 ~ file: page.tsx:29 ~ fetchData ~ user:", user);
      if (user) {
        const fetchedData = await getData(user);
        console.log("Fetched data:", fetchedData);
        setData(fetchedData);
        setConfig({
          expenseSheetId:
            fetchedData?.props.userConfig.expenseSheetId ||
            "185s3wCfiHILwWIiWieKhpJYxs4l_VO8IX1IYX_QrFtw",
          trainingRange:
            fetchedData?.props.userConfig.trainingRange ||
            "Expense-Detail!A2:F200",
          categorisationRange:
            fetchedData?.props.userConfig.categorisationRange ||
            "new_dump!A1:C200",
        });
      }
    };

    fetchData();
  }, [user]);

  const handleSpreadsheetIdChange = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    setConfig((prevConfig) => ({
      ...prevConfig,
      expenseSheetId: event.target.value,
    }));
  };

  const handleTrainingRangeChange = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    setConfig((prevConfig) => ({
      ...prevConfig,
      trainingRange: event.target.value,
    }));
  };

  const handleCategorisationRangeChange = (
    event: React.ChangeEvent<HTMLInputElement>
  ) => {
    setConfig((prevConfig) => ({
      ...prevConfig,
      categorisationRange: event.target.value,
    }));
  };

  const handleTrainClick = async () => {
    const { expenseSheetId, trainingRange } = config;
    const formData = new FormData();

    if (expenseSheetId) {
      formData.append("spreadsheetId", expenseSheetId);
    }
    if (trainingRange) {
      formData.append("range", trainingRange);
    }
    formData.append("userId", user?.sub)

    try {
      setStatusText(`Training started based on sheet ${expenseSheetId}`);
      const response = await fetch("/api/cleanAndTrain", {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        console.log("Fetched data:", data);
        setStatusText(
          `Training still running - this can take a couple of minutes`
        );
      } else {
        console.error("API call failed with status:", response.status);
      }
    } catch (error) {
      console.error("An error occurred:", error);
    }
  };

  const handleClassifyClick = async () => {
    const { expenseSheetId, categorisationRange } = config;

    const formData = new FormData();

    if (categorisationRange) {
      formData.append("range", categorisationRange);
    }
    formData.append("userId", user?.sub)
    formData.append("spreadsheetId", expenseSheetId);

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
    <ProtectedPage>
      <main className="flex flex-col min-h-screen bg-gradient-to-br from-first via-second to-third">
        <div className="flex-grow flex items-center justify-center p-10">
          <div className="w-full max-w-4xl bg-third p-6 rounded-xl shadow-lg text-white space-y-6">
            <h1 className="text-3xl font-bold leading-tight text-center">
              Step 1: Train Your Model
            </h1>
            <Instructions />
            <SpreadSheetInput
              spreadsheetLink={config.expenseSheetId}
              handleSpreadsheetLinkChange={handleSpreadsheetIdChange}
            />
            <p className="prose prose-invert">
              Range A2:F200 of sheet 'Expense Detail' is selected
            </p>
            <RangeInput
              range={config.trainingRange}
              handleRangeChange={handleTrainingRangeChange}
            />
            <div className="flex gap-3">
              <button
                onClick={handleTrainClick}
                className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
                type="button"
              >
                Train
              </button>
              <SaveConfigButton config={config} />
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
              spreadsheetLink={config.expenseSheetId}
              handleSpreadsheetLinkChange={handleSpreadsheetIdChange}
            />
            <p className="prose prose-invert">
              Range A1:C200 of sheet 'new_dump' is selected
            </p>
            <RangeInput
              range={config.categorisationRange}
              handleRangeChange={handleCategorisationRangeChange}
            />

            <div className="flex gap-3">
              <button
                onClick={handleClassifyClick}
                className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
                type="button"
              >
                Classify
              </button>
              <SaveConfigButton config={config} />
            </div>
          </div>
        </div>
      </main>
    </ProtectedPage>
  );
};

export default Demo;

async function getData(user: any) {
  if (!user) {
    return {
      props: {},
    };
  }

  const { data, error } = await supabase
    .from("account")
    .select("*")
    .eq("userId", user.sub)
    .single();

  return {
    props: {
      userConfig: data || null,
    },
  };
}
