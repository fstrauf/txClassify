import { createClient } from "@supabase/supabase-js";
import { useUser } from "@auth0/nextjs-auth0/client";
import { useState } from "react";
import toast, { Toaster } from "react-hot-toast";

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY
);

export const SaveConfigButton = ({ config }) => {
  const { user } = useUser();
  const [saveActive, setSaveActive] = useState(false);

  const handleSaveClick = async () => {
    const { expenseSheetId, trainingRange, categorisationRange } = config;
    setSaveActive(true);
    const { data, error } = await supabase
      .from("account")
      .upsert({
        userId: user.sub,
        expenseSheetId,
        trainingRange,
        categorisationRange,
      })
      .select();

    if (error) {
      console.error("Error upserting data:", error);
      setSaveActive(false);
      toast.error('Please sign in to save calculations', { position: 'bottom-right' })
    } else {
      console.log("Upserted data:", data);
      setSaveActive(false);
      toast.success('Done', { position: 'bottom-right' })
    }
  };

  return (
    <>
      <Toaster />
      <button
        onClick={handleSaveClick}
        className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded disabled:opacity-40"
        disabled={saveActive}
        type="button"
      >
        Save Config
      </button>
    </>
  );
};
