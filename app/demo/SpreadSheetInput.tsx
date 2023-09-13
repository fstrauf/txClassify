import { useAppContext } from "./DemoAppProvider";

export default function SpreadSheetInput() {
  const { config, sheetName, handleInputChange } = useAppContext();
  return (
    <label className=" flex flex-col prose prose-invert">
      Share the full url of your sheet (we'll fetch the relevant part
      automatically).
      <div className="flex items-center gap-3">
        {" "}
        <input
          type="text"
          defaultValue="185s3wCfiHILwWIiWieKhpJYxs4l_VO8IX1IYX_QrFtw"
          value={config.expenseSheetId}
          onChange={(e) => handleInputChange(e, "expenseSheetId")}
          className="mt-1 text-black w-[600px] p-1 rounded-md"
        />
        <a href={`https://docs.google.com/spreadsheets/d/${config.expenseSheetId}`} target="_blank" rel="noopener noreferrer" className="prose prose-invert text-xs">{sheetName}</a>
      </div>
    </label>
  );
}
