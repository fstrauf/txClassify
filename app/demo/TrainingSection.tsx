import React from "react";
import InstructionsTraining from "./InstructionsTraining";
import SpreadSheetInput from "./SpreadSheetInput";
import RangeInput from "./RangeInput";
import { SaveConfigButton } from "../../components/buttons/save-config-button";
import { ConfigSection } from "./ConfigSection";
import StatusText from "./statusText";
import ColumnOrderInput from "./ColumnOrderInput";
import { useAppContext } from "./DemoAppProvider";

const TrainingSection = ({}) => {
  const {
    handleInputChange,
    handleActionClick,
    config,
    setConfig,
    setTrainingStatus,
    trainingStatus,
  } = useAppContext();
  return (
    <div className="flex-grow flex items-center justify-center p-10">
      <div className="w-full max-w-4xl bg-third p-6 rounded-xl shadow-lg text-white space-y-6">
        <h1 className="text-3xl font-bold leading-tight text-center">
          Step 1: Train Your Model
        </h1>
        <InstructionsTraining />
        <ConfigSection>
          <SpreadSheetInput
          />
          <RangeInput
            handleTabChange={(e) => handleInputChange(e, "trainingTab")}
            helpText="add the name of the sheet tab that contain your already categorised expenses"
            configValue={config.trainingTab}
          />
          <ColumnOrderInput
            columns={config.columnOrderTraining}
            handleColumnsChange={(columns: any) =>
              setConfig((prevConfig: any) => ({
                ...prevConfig,
                columnOrderTraining: columns,
              }))
            }
            options={["source", "date", "description", "amount", "category"]}
            helpText='(Description and category are mandatory to have. Order the columns in the way they appear left to right in your sheet)'      
          />
          <SaveConfigButton />
        </ConfigSection>
        <div className="flex gap-3 items-center">
          <button
            onClick={() =>
              handleActionClick(
                "/api/cleanAndTrain",
                setTrainingStatus,
                // `${config.trainingTab}!${config.trainingRange}`
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
