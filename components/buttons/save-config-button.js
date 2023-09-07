import { Toaster } from "react-hot-toast";
import { useAppContext } from "../../app/demo/DemoAppProvider";

export const SaveConfigButton = () => {  
  const { saveActive, handleSaveClick } = useAppContext();

  return (
    <div className="py-2">
      <Toaster />
      <button
        onClick={handleSaveClick}
        className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded disabled:opacity-40"
        disabled={saveActive}
        type="button"
      >
        Save Config
      </button>
    </div>
  );
};
