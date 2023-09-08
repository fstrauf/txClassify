import { useAppContext } from "./DemoAppProvider";

export default function RangeInput(props: {
  handleTabChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
  helpText: string;
}) {
  const { config } = useAppContext();
  return (
    <div className="py-3">
      <label className="flex flex-col prose prose-invert">
        Sheetname
        <p className="text-xs">({props.helpText})</p>
        <div className="flex gap-1 items-center">
          <input
            type="text"
            value={config.trainingTab || "Sheet Name"}
            onChange={props.handleTabChange}
            className="mt-1 text-black p-1 rounded-md h-8"
          />
        </div>
      </label>
    </div>
  );
}
