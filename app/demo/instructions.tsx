import Link from "next/link";
import CopyableText from "./copyableText";

export default function Instructions() {
  return (
    <ol className="list-decimal ml-8 prose prose-invert mx-auto">
      <li>
        Share your sheet with{" "}
        <CopyableText textToCopy='expense-sorted@txclassify.iam.gserviceaccount.com'/>
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
    </ol>
  );
}
