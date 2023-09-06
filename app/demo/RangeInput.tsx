export default function RangeInput(props: {
  tab: string;
  range: string;
  handleTabChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
  handleRangeChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
  helpText: string;
}) {
  return (
    <div className="py-3">
      <label className="flex flex-col prose prose-invert">
        Sheetname
        <p className="text-xs">({props.helpText})</p>
        <div className="flex gap-1 items-center">
          <input
            type="text"
            value={props.tab || "Sheet Name"}
            onChange={props.handleTabChange}
            className="mt-1 text-black p-1 rounded-md h-8"
          />
          {/* <input
            type="text"
            value={props.range || "Range"}
            onChange={props.handleRangeChange}
            className="mt-1 text-black p-1 rounded-md w-32 h-8"
          /> */}
          {/* <p className="my-auto">
            = {props.tab}!{props.range}
          </p> */}
        </div>
      </label>
    </div>
  );
}
