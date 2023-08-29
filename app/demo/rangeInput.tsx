export default function RangeInput(props: {
  range: string;
  handleRangeChange: (event: React.ChangeEvent<HTMLInputElement>) => void;
}) {  

  return (
    <label className="flex flex-col">
      Selection Range:
      <input
        type="text"
        defaultValue="185s3wCfiHILwWIiWieKhpJYxs4l_VO8IX1IYX_QrFtw"
        value={props.range}
        onChange={props.handleRangeChange}
        className="mt-1 text-black w-[500px] p-1 rounded-md"
      />      
    </label>
  );
}
