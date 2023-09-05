import Link from "next/link";
import CopyableText from "./copyableText";

export default function InstructionsTraining() {
  return (
    <ol className="list-decimal ml-8 prose prose-invert mx-auto">
      <li>
        Share your sheet with{" "}
        <CopyableText textToCopy="expense-sorted@txclassify.iam.gserviceaccount.com" />
        (it's a service account that allows the script to read and write the
        classified expenses.
      </li>
      <li>
        The tool will currently only work with the{" "}
        <Link href="/fuck-you-money-sheet#instructions">
          Fuck You Money Sheet
        </Link>{" "}
        (read through instructions on how to use it)
      </li>
      <li>
        The training will take your already categorised expenses and train your
        personal model.
      </li>
      <li>
        This step won't generate any visible output just yet, but is required to
        make Step 2. the categorisation much better.
      </li>
      <li>Hit 'Train' once you have configured and downloaded the template sheet, then move to Step 2. to classify your expenses</li>
    </ol>
  );
}
