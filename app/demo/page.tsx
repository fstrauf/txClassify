"use client";
import React, { useEffect, useState } from "react";
import { createClient } from "@supabase/supabase-js";
import { useUser } from "@auth0/nextjs-auth0/client";
import toast, { Toaster } from "react-hot-toast";
import InstructionsCategorise from "./instructionsCategorise";
import InstructionsTraining from "./instructionsTraining";
import SpreadSheetInput from "./spreadSheetInput";
import RangeInput from "./rangeInput";
import ProtectedPage from "../../components/ProtectedPage";
import { SaveConfigButton } from "../../components/buttons/save-config-button";
import StatusText from "./statusText";
import ColumnOrderInput from "./ColumnOrderInput";
import ColumnMapping from "./ColumnMapping";
import { ConfigSection } from "./ConfigSection";

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL || "",
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || ""
);

export interface ConfigType {
  expenseSheetId: string;
  trainingTab: string;
  trainingRange: string;
  categorisationTab: string;
  categorisationRange: string;
  columnOrderTraining: { name: string; type: string }[];
  columnOrderCategorisation: { name: string; type: string }[];
}

const Demo = () => {
  const { user } = useUser();
  const [trainingStatus, setTrainingStatus] = useState("");
  const [categorisationStatus, setCategorisationStatus] = useState("");
  const [config, setConfig] = useState<ConfigType>({
    expenseSheetId: "185s3wCfiHILwWIiWieKhpJYxs4l_VO8IX1IYX_QrFtw",
    trainingTab: "Expense-Detail",
    trainingRange: "A2:E",
    categorisationTab: "new_dump",
    categorisationRange: "A1:C",
    columnOrderTraining: [], // default columns
    columnOrderCategorisation: [], // default columns
  });
  const [data, setData] = useState({});
  const [error, setError] = useState(null);
  const [sheetName, setSheetName] = useState("");

  useEffect(() => {
    const fetchData = async () => {
      if (user) {
        const fetchedData = await getData(user);
        console.log(
          "🚀 ~ file: page.tsx:51 ~ fetchData ~ fetchedData:",
          fetchedData
        );
        setData(fetchedData);
        setConfig({
          expenseSheetId:
            fetchedData?.props.userConfig.expenseSheetId ||
            "185s3wCfiHILwWIiWieKhpJYxs4l_VO8IX1IYX_QrFtw",
          trainingRange: fetchedData?.props.userConfig.trainingRange || "",
          trainingTab: fetchedData?.props.userConfig.trainingTab || "",
          categorisationTab:
            fetchedData?.props.userConfig.categorisationTab || "",
          categorisationRange:
            fetchedData?.props.userConfig.categorisationRange || "A1:C200",
          columnOrderTraining: fetchedData?.props.userConfig
            .columnOrderTraining || [
            { name: "Source", type: "source" },
            { name: "Date", type: "date" },
            { name: "Description", type: "description" },
            { name: "Amount", type: "amount" },
            // { name: "Credit", type: "credit" },
            { name: "Categories", type: "categories" },
          ],
          columnOrderCategorisation: fetchedData?.props.userConfig
            .columnOrderCategorisation || [
            { name: "Date", type: "date" },
            { name: "Amount", type: "amount" },
            { name: "Description", type: "description" },
          ],
        });
        getSpreadSheetData(fetchedData?.props.userConfig.expenseSheetId);
      }
    };

    fetchData();
    fetchStatus();    
  }, [user]);

  function getSpreadSheetData(expenseSheetId: string) {
    const body = { expenseSheetId };

    fetch("/api/getSpreadSheetData", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })
      .then((response) => {
        if (response.ok) {
          return response.json();
        } else {
          throw new Error(`API call failed with status: ${response.status}`);
        }
      })
      .then((data) => {
        console.log("🚀 ~ file: page.tsx:137 ~ Demo ~ data:", data);
        setSheetName(data?.spreadsheetName);
        console.log("Fetched data:", data);
      })
      .catch((error) => {
        console.error("An error occurred:", error);
      });
  }

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
            setCategorisationStatus(payload?.new?.categorisationStatus || "");
            setTrainingStatus(payload?.new?.trainingStatus || "");
          }
        )
        .subscribe();

      return () => realtimeSubscription.unsubscribe();
    } catch (error) {
      setError(error as any);
    }
  }

  const handleInputChange = async (
    event: React.ChangeEvent<HTMLInputElement>,
    field: string
  ) => {
    let value = event.target.value;
    if (field === "expenseSheetId") {
      try {
        const url = new URL(value);
        value = url.pathname.split("/")[3];
        if (value !== config.expenseSheetId) {
          getSpreadSheetData(value)
        }
      } catch (error) {
        console.error("Invalid URL:", error);
      }
    }
    setConfig((prevConfig) => ({
      ...prevConfig,
      [field]: value,
    }));
  };

  const handleActionClick = async (
    apiUrl: string,
    statusSetter: Function,
    range: string
  ) => {
    const { expenseSheetId } = config || {};
    const formData = new FormData();
    // Check if file exists in Supabase bucket
    statusSetter(`Action started based on sheet ${expenseSheetId}`);

    if (apiUrl === "/api/cleanAndClassify") {
      const response = await supabase.storage.from("txclassify").list("", {
        limit: 2,
        offset: 0,
        sortBy: { column: "name", order: "asc" },
        search: expenseSheetId,
      });
      if (error) {
        console.error("Failed to retrieve file list:", error);
        toast.error("Can't find training data, please run training first", {
          position: "bottom-right",
        });
        statusSetter("");
        return;
      }
      if (response?.data?.length === 0) {
        toast.error("Can't find training data, please run training first", {
          position: "bottom-right",
        });
        statusSetter("");
        return;
      }
    }

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
        statusSetter("");
      }
    } catch (error) {
      console.error("An error occurred:", error);
      statusSetter("");
    }
  };

  return (
    <ProtectedPage>
      <Toaster />
      <main className="flex flex-col min-h-screen bg-gradient-to-br from-first via-second to-third">
        <div className="flex-grow flex items-center justify-center p-10">
          <div className="w-full max-w-4xl bg-third p-6 rounded-xl shadow-lg text-white space-y-6">
            <h1 className="text-3xl font-bold leading-tight text-center">
              Step 1: Train Your Model
            </h1>
            <InstructionsTraining />
            <ConfigSection>
              <SpreadSheetInput
                spreadsheetLink={config.expenseSheetId}
                handleSpreadsheetLinkChange={(e) =>
                  handleInputChange(e, "expenseSheetId")
                }
                sheetName={sheetName}
              />
              <RangeInput
                tab={config.trainingTab}
                range={config.trainingRange}
                handleTabChange={(e) => handleInputChange(e, "trainingTab")}
                handleRangeChange={(e) => handleInputChange(e, "trainingRange")}
                helpText="add the name of the sheet and the range that covers the columns Source, Date, Description, Amount, Category of your already categorised expenses"
              />
              <SaveConfigButton config={config} />
            </ConfigSection>
            {/* <ColumnOrderInput
              columns={config.columnOrderTraining}
              handleColumnsChange={(columns: any) =>
                setConfig((prevConfig) => ({
                  ...prevConfig,
                  columnOrderTraining: columns,
                }))
              }
            /> */}
            {/* <ColumnMapping
              googleSheetColumns={config.columnOrderTraining}
              expectedColumns={[
                "Source",
                "Date",
                "Narrative",
                "Debit",
                "Credit",
                "Categories",
              ]}
            /> */}
            <div className="flex gap-3 items-center">
              <button
                onClick={() =>
                  handleActionClick(
                    "/api/cleanAndTrain",
                    setTrainingStatus,
                    `${config.trainingTab}!${config.trainingRange}`
                  )
                }
                className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded disabled:opacity-40"
                type="button"
                disabled={
                  trainingStatus !== "" && trainingStatus !== "completed"
                }
              >
                Train
              </button>

              <StatusText text={trainingStatus} />
            </div>
          </div>
        </div>
        {/* user flow:
1. upload the spreadsheet, 
2. read the columns, 
3. let the user assign the type of each column. 

more basic:

1. let the user add the type of his columns*/}
        <div className="flex-grow flex items-center justify-center p-10">
          <div className="w-full max-w-4xl bg-third p-6 rounded-xl shadow-lg text-white space-y-6">
            <h1 className="text-3xl font-bold leading-tight text-center">
              Step 2: Classify your Expenses
            </h1>
            <InstructionsCategorise trainingTab={config.trainingTab} />
            <ConfigSection>
              <SpreadSheetInput
                spreadsheetLink={config.expenseSheetId}
                handleSpreadsheetLinkChange={(e) =>
                  handleInputChange(e, "expenseSheetId")
                }
              />
              <RangeInput
                tab={config.categorisationTab}
                range={config.categorisationRange}
                handleTabChange={(e) =>
                  handleInputChange(e, "categorisationTab")
                }
                handleRangeChange={(e) =>
                  handleInputChange(e, "categorisationRange")
                }
                helpText="add the name of the sheet and the range that covers the columns Date, Description, Amount of the expenses you want to categorise"
              />
              <SaveConfigButton config={config} />
            </ConfigSection>

            <div className="flex gap-3 items-center">
              <button
                onClick={() =>
                  handleActionClick(
                    "/api/cleanAndClassify",
                    setCategorisationStatus,
                    `${config.categorisationTab}!${config.categorisationRange}`
                  )
                }
                className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded disabled:opacity-40"
                type="button"
                disabled={
                  categorisationStatus !== "" &&
                  categorisationStatus !== "completed"
                }
              >
                Classify
              </button>
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
