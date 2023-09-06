import React, { useState } from "react";
import InstructionsCategorise from "./InstructionsCategorise";
import SpreadSheetInput from "./SpreadSheetInput";
import RangeInput from "./RangeInput";
import { SaveConfigButton } from "../../components/buttons/save-config-button";
import { ConfigSection } from "./ConfigSection";
import StatusText from "./statusText";
import { ConfigType } from "./page";
import ColumnOrderInput from "./ColumnOrderInput";

interface ClassificationSectionProps {
  config: ConfigType;
  setConfig: React.Dispatch<React.SetStateAction<ConfigType>>;
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
  categorisationStatus: string;
  setCategorisationStatus: React.Dispatch<React.SetStateAction<string>>;
}

const ClassificationSection: React.FC<ClassificationSectionProps> = ({
  config,
  setConfig,
  handleInputChange,
  handleActionClick,
  sheetName,
  categorisationStatus,
  setCategorisationStatus,
}) => {
  // const [categorisationStatus, setCategorisationStatus] = useState("");

  return (
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
            sheetName={sheetName}
          />
          <RangeInput
            tab={config.categorisationTab}
            range={config.categorisationRange}
            handleTabChange={(e) => handleInputChange(e, "categorisationTab")}
            handleRangeChange={(e) =>
              handleInputChange(e, "categorisationRange")
            }
            helpText="add the name of the sheet and the range that covers the columns Date, Description, Amount of the expenses you want to categorise"
          />
          <ColumnOrderInput
            columns={config.columnOrderCategorisation}
            handleColumnsChange={(columns: any) =>
              setConfig((prevConfig: any) => ({
                ...prevConfig,
                columnOrderCategorisation: columns,
              }))
            }
            options={["date", "description", "amount"]}
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
  );
};

export default ClassificationSection;
