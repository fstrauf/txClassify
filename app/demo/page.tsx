"use client";
import React, { useEffect, useState } from "react";
import { createClient } from "@supabase/supabase-js";
import { useUser } from "@auth0/nextjs-auth0/client";
import toast, { Toaster } from "react-hot-toast";
import ProtectedPage from "../../components/ProtectedPage";
import TrainingSection from "./TrainingSection";
import ClassificationSection from "./ClassificationSection";
import { AppProvider, useAppContext } from "./DemoAppProvider";

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
  columnOrderTraining: { name: string; type: string; index: number }[];
  columnOrderCategorisation: { name: string; type: string; index: number }[];
}

const Demo = () => {
  // const { user } = useUser();
  // const [trainingStatus, setTrainingStatus] = useState("");
  // const [categorisationStatus, setCategorisationStatus] = useState("");
  // const [config, setConfig] = useState<ConfigType>({
  //   expenseSheetId: "185s3wCfiHILwWIiWieKhpJYxs4l_VO8IX1IYX_QrFtw",
  //   trainingTab: "Expense-Detail",
  //   trainingRange: "A2:E",
  //   categorisationTab: "new_dump",
  //   categorisationRange: "A1:C",
  //   columnOrderTraining: [], // default columns
  //   columnOrderCategorisation: [], // default columns
  // });
  // const [data, setData] = useState({});
  // const [error, setError] = useState(null);
  // const [sheetName, setSheetName] = useState("");

  // const { handleInputChange, handleActionClick, config, setConfig, sheetName, setSheetName } = useAppContext();
  

  // useEffect(() => {
  //   const fetchData = async () => {
  //     if (user) {
  //       const fetchedData = await getData(user);

  //       setData(fetchedData);
  //       setConfig({
  //         expenseSheetId:
  //           fetchedData?.props.userConfig.expenseSheetId ||
  //           "185s3wCfiHILwWIiWieKhpJYxs4l_VO8IX1IYX_QrFtw",
  //         trainingRange: fetchedData?.props.userConfig.trainingRange || "",
  //         trainingTab: fetchedData?.props.userConfig.trainingTab || "",
  //         categorisationTab:
  //           fetchedData?.props.userConfig.categorisationTab || "",
  //         categorisationRange:
  //           fetchedData?.props.userConfig.categorisationRange || "A1:C200",
  //         columnOrderTraining: fetchedData?.props.userConfig
  //           .columnOrderTraining || [
  //           { name: "A", type: "source", index: 1 },
  //           { name: "B", type: "date", index: 2 },
  //           { name: "C", type: "description", index: 3 },
  //           { name: "D", type: "amount", index: 4 },
  //           { name: "E", type: "categories", index: 5 },
  //         ],
  //         columnOrderCategorisation: fetchedData?.props.userConfig
  //           .columnOrderCategorisation || [
  //           { name: "A", type: "date", index: 1 },
  //           { name: "B", type: "amount", index: 2 },
  //           { name: "C", type: "description", index: 3 },
  //         ],
  //       });
  //       getSpreadSheetData(fetchedData?.props.userConfig.expenseSheetId);
  //     }
  //   };

  //   fetchData();
  //   fetchStatus();
  // }, [user]);

  // function getSpreadSheetData(expenseSheetId: string) {
  //   const body = { expenseSheetId };

  //   fetch("/api/getSpreadSheetData", {
  //     method: "POST",
  //     headers: { "Content-Type": "application/json" },
  //     body: JSON.stringify(body),
  //   })
  //     .then((response) => {
  //       if (response.ok) {
  //         return response.json();
  //       } else {
  //         throw new Error(`API call failed with status: ${response.status}`);
  //       }
  //     })
  //     .then((data) => {
  //       console.log("🚀 ~ file: page.tsx:137 ~ Demo ~ data:", data);
  //       setSheetName(data?.spreadsheetName);
  //       console.log("Fetched data:", data);
  //     })
  //     .catch((error) => {
  //       console.error("An error occurred:", error);
  //     });
  // }

  // async function fetchStatus() {
  //   try {
  //     const realtimeSubscription = supabase
  //       .channel("any")
  //       .on(
  //         "postgres_changes",
  //         {
  //           event: "UPDATE",
  //           schema: "public",
  //           table: "account",
  //           filter: `userId=eq.${user?.sub}`,
  //         },
  //         (payload) => {
  //           setCategorisationStatus(payload?.new?.categorisationStatus || "");
  //           setTrainingStatus(payload?.new?.trainingStatus || "");
  //         }
  //       )
  //       .subscribe();

  //     return () => realtimeSubscription.unsubscribe();
  //   } catch (error) {
  //     setError(error as any);
  //   }
  // }

  // const handleInputChange = async (
  //   event: React.ChangeEvent<HTMLInputElement>,
  //   field: string
  // ) => {
  //   let value = event.target.value;
  //   if (field === "expenseSheetId") {
  //     try {
  //       const url = new URL(value);
  //       value = url.pathname.split("/")[3];
  //       if (value !== config.expenseSheetId) {
  //         getSpreadSheetData(value);
  //       }
  //     } catch (error) {
  //       console.error("Invalid URL:", error);
  //     }
  //   }
  //   setConfig((prevConfig) => ({
  //     ...prevConfig,
  //     [field]: value,
  //   }));
  // };

  // const handleActionClick = async (apiUrl: string, statusSetter: Function) => {
  //   const { expenseSheetId, columnOrderTraining, trainingTab } = config || {};

  //   statusSetter(`Action started based on sheet ${expenseSheetId}`);

  //   if (apiUrl === "/api/cleanAndClassify") {
  //     const response = await supabase.storage.from("txclassify").list("", {
  //       limit: 2,
  //       offset: 0,
  //       sortBy: { column: "name", order: "asc" },
  //       search: expenseSheetId,
  //     });
  //     if (error) {
  //       console.error("Failed to retrieve file list:", error);
  //       toast.error("Can't find training data, please run training first", {
  //         position: "bottom-right",
  //       });
  //       statusSetter("");
  //       return;
  //     }
  //     if (response?.data?.length === 0) {
  //       toast.error("Can't find training data, please run training first", {
  //         position: "bottom-right",
  //       });
  //       statusSetter("");
  //       return;
  //     }
  //   }

  //   const userId = user?.sub;
  //   const body = { expenseSheetId, columnOrderTraining, userId, trainingTab };

  //   try {
  //     statusSetter(`Action started based on sheet ${expenseSheetId}`);
  //     const response = await fetch(apiUrl, {
  //       method: "POST",
  //       headers: { "Content-Type": "application/json" },
  //       body: JSON.stringify(body),
  //     });

  //     if (response.ok) {
  //       const data = await response.json();
  //       console.log("Fetched data:", data);
  //     } else {
  //       console.error("API call failed with status:", response.status);
  //       statusSetter("");
  //     }
  //   } catch (error) {
  //     console.error("An error occurred:", error);
  //     statusSetter("");
  //   }
  // };

  return (
    <ProtectedPage>
      <AppProvider>
        <Toaster />
        <main className="flex flex-col min-h-screen bg-gradient-to-br from-first via-second to-third">
          <TrainingSection
            // config={config}
            // setConfig={setConfig}
            // handleInputChange={handleInputChange}
            // handleActionClick={handleActionClick}
            // sheetName={sheetName}
            // trainingStatus={trainingStatus}
            // setTrainingStatus={setTrainingStatus}
          />
          <ClassificationSection
            // config={config}
            // setConfig={setConfig}
            // handleInputChange={handleInputChange}
            // handleActionClick={handleActionClick}
            // sheetName={sheetName}
            // categorisationStatus={categorisationStatus}
            // setCategorisationStatus={setCategorisationStatus}
          />
        </main>
      </AppProvider>
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
