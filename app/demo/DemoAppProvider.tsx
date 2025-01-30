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
      { name: "F", type: "currency", index: 6 },
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
  const [error, setError] = useState<string | null>(null);
  const [sheetName, setSheetName] = useState("");
  const [saveActive, setSaveActive] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      if (user) {
        console.log("Starting fetchData with user:", user);
        try {
          const fetchedData = await getData(user);
          console.log("Fetched data result:", fetchedData);
          
          if (fetchedData.props.error) {
            console.error("Error loading user configuration:", fetchedData.props.error);
            setError(fetchedData.props.error);
            return;
          }

          setData(fetchedData);

          // Only use default config if no user config exists
          if (fetchedData?.props?.userConfig) {
            console.log("Found user config:", fetchedData.props.userConfig);
            const userConfig = fetchedData.props.userConfig;
            const newConfig = {
              expenseSheetId: userConfig.expenseSheetId,
              trainingRange: userConfig.trainingRange || configDefault.trainingRange,
              trainingTab: userConfig.trainingTab || configDefault.trainingTab,
              categorisationTab: userConfig.categorisationTab || configDefault.categorisationTab,
              categorisationRange: userConfig.categorisationRange || configDefault.categorisationRange,
              columnOrderTraining: userConfig.columnOrderTraining || configDefault.columnOrderTraining,
              columnOrderCategorisation: userConfig.columnOrderCategorisation || configDefault.columnOrderCategorisation,
            };
            console.log("Setting new config:", newConfig);
            setConfig(newConfig);
            
            // Fetch spreadsheet data with user's sheet ID
            console.log("Fetching spreadsheet data with ID:", userConfig.expenseSheetId);
            getSpreadSheetData(userConfig.expenseSheetId);
          } else {
            console.log("No user configuration found, using defaults:", configDefault);
            setConfig(configDefault);
            getSpreadSheetData(configDefault.expenseSheetId);
          }
        } catch (err) {
          console.error("Error in fetchData:", err);
          setError(err as any);
        }
      } else {
        console.log("No user available in fetchData");
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
        userId,
        sheetId: expenseSheetId,
        files: [{
          config: {
            column_mapping: {
              date: "0",
              amount: "1",
              description: "2"
            },
            tab: categorisationTab,
            range: config.categorisationRange || "A1:C",
            defaultCurrency: "AUD",
            date_format: "%d/%m/%Y"
          },
          type: "google_sheets",
          path: expenseSheetId
        }]
      };
    } else {
      body = {
        expenseSheetId,
        columnOrderTraining,
        userId,
        trainingTab,
      };
    }

    try {
      const response = await fetch(apiUrl, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || "An error occurred");
      }

      const data = await response.json();
      console.log("Action completed successfully:", data);
    } catch (error) {
      console.error("Error during action:", error);
      throw error;
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
  console.log("🚀 ~ getData ~ user:", user)
  if (!user) {
    console.log("No user provided to getData");
    return {
      props: {},
    };
  }

  console.log("Fetching data for user ID:", user.sub);
  
  // First, let's see all accounts with full details
  const { data: allAccounts, error: listError } = await supabase
    .from("account")
    .select("*");
    
  console.log("All accounts in database (full details):", allAccounts);

  // Also try to find any accounts that might match the email
  const { data: emailAccounts, error: emailError } = await supabase
    .from("account")
    .select("*")
    .eq("email", user.email);
    
  console.log("Accounts matching email:", emailAccounts);

  const { data, error } = await supabase
    .from("account")
    .select("*")
    .eq("userId", user.sub)
    .single();

  if (error) {
    console.error("Error fetching user config:", error);
    return {
      props: {
        error: error.message,
      },
    };
  }

  console.log("Raw data from database:", data);

  // Parse JSON strings for column orders if they exist
  if (data) {
    try {
      if (typeof data.columnOrderTraining === 'string') {
        console.log("Parsing columnOrderTraining:", data.columnOrderTraining);
        data.columnOrderTraining = JSON.parse(data.columnOrderTraining.replace(/\\"/g, '"'));
      }
      if (typeof data.columnOrderCategorisation === 'string') {
        console.log("Parsing columnOrderCategorisation:", data.columnOrderCategorisation);
        data.columnOrderCategorisation = JSON.parse(data.columnOrderCategorisation.replace(/\\"/g, '"'));
      }
    } catch (e) {
      console.error("Error parsing column order JSON:", e);
    }
  }

  console.log("Processed data:", data);
  return {
    props: {
      userConfig: data || null,
    },
  };
}
