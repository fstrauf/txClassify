export interface SpreadSheetInputProps {
  spreadsheetLink: string;
  handleSpreadsheetLinkChange: (
    event: React.ChangeEvent<HTMLInputElement>
  ) => void;
  sheetName: string;
}

export default function SpreadSheetInput(props: SpreadSheetInputProps) {
  return (
    <label className=" flex flex-col prose prose-invert">
      Share the full url of your sheet (we'll fetch the relevant part
      automatically).
      <div className="flex items-center gap-3">
        {" "}
        <input
          type="text"
          defaultValue="185s3wCfiHILwWIiWieKhpJYxs4l_VO8IX1IYX_QrFtw"
          value={props.spreadsheetLink}
          onChange={props.handleSpreadsheetLinkChange}
          className="mt-1 text-black w-[500px] p-1 rounded-md"
        />
        <p className="prose prose-invert text-xs">= {props.sheetName}</p>
      </div>
    </label>
  );
}