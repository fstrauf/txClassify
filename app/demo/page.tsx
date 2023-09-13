"use client";
import React, { Component, ErrorInfo, ReactNode } from "react";
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

interface ErrorBoundaryProps {
  children: ReactNode;
}

class ErrorBoundary extends Component<ErrorBoundaryProps> {
  state = { hasError: false, error: null, errorInfo: null };

  static getDerivedStateFromError(error: Error) {
    return { hasError: true };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error("Uncaught error:", error, errorInfo);
    this.setState({ error, errorInfo });
  }

  render() {
    if (this.state.hasError) {
      return <h1>Sorry, something went wrong.</h1>;
    }

    return this.props?.children;
  }
}

const Demo = () => {
  return (
    <ProtectedPage>
      <AppProvider>
        <Toaster />
        <ErrorBoundary>
          <main className="flex flex-col min-h-screen bg-gradient-to-br from-first via-second to-third">
            <TrainingSection />
            <ClassificationSection />
          </main>
        </ErrorBoundary>
      </AppProvider>
    </ProtectedPage>
  );
};

export default Demo;