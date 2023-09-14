"use client";
import React from "react";
import { Toaster } from "react-hot-toast";
import ProtectedPage from "../../components/ProtectedPage";
import TrainingSection from "./TrainingSection";
import ClassificationSection from "./ClassificationSection";
import { AppProvider } from "./DemoAppProvider";

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