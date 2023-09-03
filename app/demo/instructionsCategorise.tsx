import Link from "next/link";
import CopyableText from "./copyableText";

interface InstructionsCategoriseProps {
  trainingTab: string;
}

export default function InstructionsCategorise({ trainingTab }: InstructionsCategoriseProps) {
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
        Categorising will use your trained expenses (run training first!) based
        on the training
      </li>
      <li>
        All expenses from the tab configured below will have categories added.
        The result will be added to your main tab (the one configured in the
        training section)
      </li>
      <li>Once completed, head over to the {trainingTab} tab to review and adjust the categories.</li>
    </ol>
  );
}