import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  ReactNode,
} from "react";
import { createClient } from "@supabase/supabase-js";
import { useUser } from "@auth0/nextjs-auth0/client";
import toast from "react-hot-toast";

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL || "",
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || ""
);

interface AppContextType {
  user: any;
  trainingStatus: string;
  setTrainingStatus: React.Dispatch<React.SetStateAction<string>>;
  categorisationStatus: string;
  setCategorisationStatus: React.Dispatch<React.SetStateAction<string>>;
  config: ConfigType;
  setConfig: React.Dispatch<React.SetStateAction<ConfigType>>;
  handleInputChange: (
    event: React.ChangeEvent<HTMLInputElement>,
    field: string
  ) => Promise<void>;
  handleActionClick: (apiUrl: string, statusSetter: Function) => Promise<void>;
  sheetName: string;
  setSheetName: React.Dispatch<React.SetStateAction<string>>;
  saveActive: boolean;
  setSaveActive: React.Dispatch<React.SetStateAction<boolean>>;
  handleSaveClick: () => Promise<void>;
}

// Create a context
const AppContext = createContext<AppContextType | undefined>(undefined);

// Create a provider component
interface AppProviderProps {
  children: ReactNode;
}

export interface ConfigType {
  expenseSheetId: string;
  trainingTab: string;
  trainingRange: string;
  categorisationTab: string;
  categorisationRange: string;
  columnOrderTraining: { name: string; type: string; index: number }[];
  columnOrderCategorisation: { name: string; type: string; index: number }[];
}

export const AppProvider = ({ children }: AppProviderProps) => {
  const configDefault = {
    expenseSheetId: "1Buon6FEg7JGJMjuZgNgIrm5XyfP38JeaOJTNv6YQSHA",
    trainingTab: "Expense-Detail",
    trainingRange: "A2:E",
    categorisationTab: "new_dump",
    categorisationRange: "A1:C",
    columnOrderTraining: [
      { name: "A", type: "source", index: 1 },
      { name: "B", type: "date", index: 2 },
      { name: "C", type: "description", index: 3 },
      { name: "D", type: "amount", index: 4 },
      { name: "E", type: "category", index: 5 },
    ], // default columns
    columnOrderCategorisation: [
      { name: "A", type: "date", index: 1 },
      { name: "B", type: "amount", index: 2 },
      { name: "C", type: "description", index: 3 },
    ], // default columns
  };
  const { user } = useUser();
  const [trainingStatus, setTrainingStatus] = useState("");
  const [categorisationStatus, setCategorisationStatus] = useState("");
  const [config, setConfig] = useState<ConfigType>(configDefault);
  const [data, setData] = useState({});
  const [error, setError] = useState(null);
  const [sheetName, setSheetName] = useState("");
  const [saveActive, setSaveActive] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      if (user) {
        const fetchedData = await getData(user);

        setData(fetchedData);

        setConfig({
          expenseSheetId:
            fetchedData?.props?.userConfig?.expenseSheetId ||
            configDefault.expenseSheetId,
          trainingRange: "",
          trainingTab:
            fetchedData?.props?.userConfig?.trainingTab ||
            configDefault.trainingTab,
          categorisationTab:
            fetchedData?.props?.userConfig?.categorisationTab ||
            configDefault.categorisationTab,
          categorisationRange: "",
          columnOrderTraining:
            fetchedData?.props?.userConfig?.columnOrderTraining ||
            configDefault.columnOrderTraining,
          columnOrderCategorisation:
            fetchedData?.props?.userConfig?.columnOrderCategorisation ||
            configDefault.columnOrderCategorisation,
        });
        getSpreadSheetData(
          fetchedData?.props?.userConfig?.expenseSheetId ||
            configDefault.expenseSheetId
        );
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
        } else if (response.status === 401) {
          setSheetName("Not authorised to open sheet, please add technical user");
          throw new Error('Unauthorized: You do not have access to this sheet.');
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
    console.log("🚀 ~ file: DemoAppProvider.tsx:178 ~ AppProvider ~ field:", field)
    let value = event.target.value;
    console.log("🚀 ~ file: DemoAppProvider.tsx:179 ~ AppProvider ~ value:", value)
    if (field === "expenseSheetId") {
      try {
        const url = new URL(value);
        value = url.pathname.split("/")[3];
        if (value !== config.expenseSheetId) {
          getSpreadSheetData(value);
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

  const handleActionClick = async (apiUrl: string, statusSetter: Function) => {
    const {
      expenseSheetId,
      columnOrderTraining,
      trainingTab,
      categorisationTab,
      columnOrderCategorisation,
    } = config || {};

    const userId = user?.sub;
    if (!userId) {
      console.log("Needs to be logged on to run training or categorisation");
      toast.error("You need to be logged in to run this", {
        position: "bottom-right",
      });
      return;
    }
    let body = {};

    if (apiUrl === "/api/cleanAndClassify") {
      updateProcessStatus(
        `Action started based on sheet ${expenseSheetId}`,
        "categorisationStatus",
        userId
      );
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
      body = {
        expenseSheetId,
        columnOrderCategorisation,
        userId,
        categorisationTab,
      };
    } else {
      body = { expenseSheetId, columnOrderTraining, userId, trainingTab };
      updateProcessStatus(
        `Action started based on sheet ${expenseSheetId}`,
        "trainingStatus",
        userId
      );
    }

    handleSaveClick();

    try {
      statusSetter(`Action started based on sheet ${expenseSheetId}`);
      const response = await fetch(apiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
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

  const updateProcessStatus = async (
    status_text: string,
    mode: string,
    userId: string
  ) => {
    const updateObject: { [key: string]: string } = {};
    updateObject[mode] = status_text;
    const { data, error } = await supabase
      .from("account")
      .update(updateObject)
      .eq("userId", userId)
      .select();

    if (error) {
      console.error("Error upserting data:", error);
    } else {
      console.log("Upserted data:", data);
    }
  };

  const handleSaveClick = async () => {
    const {
      expenseSheetId,
      trainingRange,
      categorisationRange,
      trainingTab,
      categorisationTab,
      columnOrderTraining,
      columnOrderCategorisation,
    } = config;
    setSaveActive(true);
    const { data, error } = await supabase
      .from("account")
      .upsert({
        userId: user?.sub,
        expenseSheetId,
        trainingRange,
        categorisationRange,
        categorisationTab,
        trainingTab,
        columnOrderTraining,
        columnOrderCategorisation,
      })
      .select();

    if (error) {
      console.error("Error upserting data:", error);
      setSaveActive(false);
      toast.error("Please sign in to save calculations", {
        position: "bottom-right",
      });
    } else {
      console.log("Upserted data:", data);
      setSaveActive(false);
      toast.success("Configuration Saved!", { position: "bottom-right" });
    }
  };

  return (
    <AppContext.Provider
      value={{
        user,
        trainingStatus,
        setTrainingStatus,
        categorisationStatus,
        setCategorisationStatus,
        config,
        setConfig,
        handleInputChange,
        handleActionClick,
        sheetName,
        setSheetName,
        saveActive,
        setSaveActive,
        handleSaveClick,
      }}
    >
      {children}
    </AppContext.Provider>
  );
};

// Create a hook to use the context
export const useAppContext = () => {
  const context = useContext(AppContext);
  if (context === undefined) {
    throw new Error("useAppContext must be used within an AppProvider");
  }
  return context;
};

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
