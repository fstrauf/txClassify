import { createClient } from "@supabase/supabase-js";
import { useUser } from "@auth0/nextjs-auth0/client";

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY
);

export const SaveConfigButton = ({ config }) => {
  const { user } = useUser();

  const handleSaveClick = async () => {
    const { expenseSheetId, dataTabTraining, dataTabClassify } = config;

    const { data, error } = await supabase
      .from("account")
      .upsert({
        userId: user.sub,
        expenseSheetId,
        dataTabTraining,
        dataTabClassify,
      })
      .select();

    if (error) {
      console.error("Error upserting data:", error);
    } else {
      console.log("Upserted data:", data);
    }
  };

  return (
    <button
      onClick={handleSaveClick}
      className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
    >
      Save Config
    </button>
  );
};
