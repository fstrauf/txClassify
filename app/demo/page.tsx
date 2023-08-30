"use client";
import React, { useEffect, useState } from "react";
import { createClient } from "@supabase/supabase-js";
import { useUser } from "@auth0/nextjs-auth0/client";

import Instructions from "./instructions";
import SpreadSheetInput from "./spreadSheetInput";
import RangeInput from "./rangeInput";
import ProtectedPage from "../../components/ProtectedPage";
import { SaveConfigButton } from "../../components/buttons/save-config-button";
import StatusText from "./statusText";

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL || "",
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || ""
);

interface ConfigType {
  expenseSheetId: string;
  trainingRange: string;
  categorisationRange: string;
}

const Demo = () => {
  const { user } = useUser();
  const [trainingStatus, setTrainingStatus] = useState("");
  const [categorisationStatus, setCategorisationStatus] = useState("");
  const [config, setConfig] = useState<ConfigType>({
    expenseSheetId: "",
    trainingRange: "",
    categorisationRange: "",
  });
  const [data, setData] = useState({});
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchData = async () => {
      if (user) {
        const fetchedData = await getData(user);
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
    fetchStatus();
  }, [user]);

  async function fetchStatus() {
    try {
      const realtimeSubscription = supabase
        .channel("any")
        .on(
          "postgres_changes",
          {
            event: "UPDATE",
            schema: "public",
            table: "account",
            filter: `userId=eq.${user?.sub}`,
          },
          (payload) => {
            setCategorisationStatus(payload?.new?.categorisationStatus || '');
            setTrainingStatus(payload?.new?.trainingStatus || '');
          }
        )
        .subscribe();

      return () => realtimeSubscription.unsubscribe();
    } catch (error) {
      setError(error as any);
    }
  }

  const handleInputChange = (event: React.ChangeEvent<HTMLInputElement>, field: string) => {
    setConfig((prevConfig) => ({
      ...prevConfig,
      [field]: event.target.value,
    }));
  };

  const handleActionClick = async (apiUrl: string, statusSetter: Function, range: string) => {
    const { expenseSheetId } = config || {};
    const formData = new FormData();

    if (expenseSheetId) {
      formData.append("spreadsheetId", expenseSheetId);
    }
    if (range) {
      formData.append("range", range);
    }
    const userId = user?.sub;
    if (userId) {
      formData.append("userId", userId);
    }

    try {
      statusSetter(`Action started based on sheet ${expenseSheetId}`);
      const response = await fetch(apiUrl, {
        method: "POST",
        body: formData,
      });

      if (response.ok) {
        const data = await response.json();
        console.log("Fetched data:", data);
      } else {
        console.error("API call failed with status:", response.status);
        statusSetter('')
      }
    } catch (error) {
      console.error("An error occurred:", error);
      statusSetter('')
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
              handleSpreadsheetLinkChange={(e) => handleInputChange(e, 'expenseSheetId')}
            />
            <p className="prose prose-invert">
              Range A2:F200 of sheet 'Expense Detail' is selected
            </p>
            <RangeInput
              range={config.trainingRange}
              handleRangeChange={(e) => handleInputChange(e, 'trainingRange')}
            />
            <div className="flex gap-3">
              <button
                onClick={() => handleActionClick("/api/cleanAndTrain", setTrainingStatus, config.trainingRange)}
                className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded disabled:opacity-40"
                type="button"
                disabled={
                  trainingStatus !== "" && trainingStatus !== "completed"
                }
              >
                Train
              </button>

              <SaveConfigButton config={config} />
              <StatusText text={trainingStatus} />
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
              handleSpreadsheetLinkChange={(e) => handleInputChange(e, 'expenseSheetId')}
            />
            <p className="prose prose-invert">
              Range A1:C200 of sheet 'new_dump' is selected
            </p>
            <RangeInput
              range={config.categorisationRange}
              handleRangeChange={(e) => handleInputChange(e, 'categorisationRange')}
            />

            <div className="flex gap-3">
              <button
                onClick={() => handleActionClick("/api/cleanAndClassify", setCategorisationStatus, config.categorisationRange)}
                className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded disabled:opacity-40"
                type="button"
                disabled={
                  categorisationStatus !== "" &&
                  categorisationStatus !== "completed"
                }
              >
                Classify
              </button>
              <SaveConfigButton config={config} />
              <StatusText text={categorisationStatus} />
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

