"use client";
import React from "react";
import { Toaster } from "react-hot-toast";
import ProtectedPage from "../../components/ProtectedPage";
import TrainingSection from "./TrainingSection";
import ClassificationSection from "./ClassificationSection";
import { AppProvider } from "./DemoAppProvider";

// const supabase = createClient(
//   process.env.NEXT_PUBLIC_SUPABASE_URL || "",
//   process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY || ""
// );

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
  return (
    <ProtectedPage>
      <AppProvider>
        <Toaster />
        <main className="flex flex-col min-h-screen bg-gradient-to-br from-first via-second to-third">
          <TrainingSection />
          <ClassificationSection />
        </main>
      </AppProvider>
    </ProtectedPage>
  );
};

export default Demo;

// async function getData(user: any) {
//   if (!user) {
//     return {
//       props: {},
//     };
//   }

//   const { data, error } = await supabase
//     .from("account")
//     .select("*")
//     .eq("userId", user.sub)
//     .single();

//   return {
//     props: {
//       userConfig: data || null,
//     },
//   };
// }
