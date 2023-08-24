import Image from "next/image";

export default function SpreadSheetInput(props: {
  spreadsheetLink: string;
  handleSpreadsheetLinkChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
}) {
  console.log("🚀 ~ spreadsheetLink:", props.spreadsheetLink);

  return (
    <label className="block">
      Spreadsheet ID:
      <div className="m-6">
        <Image
          width={1620 / 2.5}
          height={82 / 2.5}
          src="/f-you-sheet-id.png"
          className="rounded-md"
          alt="Add your income to the sheet"
        />
      </div>
      <input
        type="text"
        defaultValue="185s3wCfiHILwWIiWieKhpJYxs4l_VO8IX1IYX_QrFtw"
        value={props.spreadsheetLink}
        onChange={props.handleSpreadsheetLinkChange}
        className="mt-1 text-black w-[500px] p-1 rounded-md"
      />      
    </label>
  );
}
