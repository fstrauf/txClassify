import React, { useState } from "react";
import InstructionsTraining from "./InstructionsTraining";
import SpreadSheetInput from "./SpreadSheetInput";
import RangeInput from "./RangeInput";
import { SaveConfigButton } from "../../components/buttons/save-config-button";
import { ConfigSection } from "./ConfigSection";
import StatusText from "./statusText";
import { ConfigType } from "./page";

interface TrainingSectionProps {
  config: ConfigType;
  handleInputChange: (
    event: React.ChangeEvent<HTMLInputElement>,
    field: string
  ) => void;
  handleActionClick: (
    apiUrl: string,
    statusSetter: Function,
    range: string
  ) => void;
  sheetName: string;
}

const TrainingSection: React.FC<TrainingSectionProps> = ({
  config,
  handleInputChange,
  handleActionClick,
  sheetName,
}) => {
  const [trainingStatus, setTrainingStatus] = useState("");

  return (
    <div className="flex-grow flex items-center justify-center p-10">
      <div className="w-full max-w-4xl bg-third p-6 rounded-xl shadow-lg text-white space-y-6">
        {/* user flow:
1. upload the spreadsheet, 
2. read the columns, 
3. let the user assign the type of each column. 

more basic:

1. let the user add the type of his columns*/}
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
            disabled={trainingStatus !== "" && trainingStatus !== "completed"}
          >
            Train
          </button>

          <StatusText text={trainingStatus} />
        </div>
      </div>
    </div>
  );
};

export default TrainingSection;