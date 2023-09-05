export default function SpreadSheetInput(props: {
  spreadsheetLink: string;
  handleSpreadsheetLinkChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
}) {

  return (
    <label className=" flex flex-col prose prose-invert">
      Share the full url of your sheet (we'll fetch the relevant part automatically).
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
