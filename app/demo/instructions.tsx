import Link from "next/link";

export default function Instructions() {
  return (
    <ol className="list-decimal ml-8 prose prose-invert mx-auto">
      <li>
        Share your sheet with{" "}
        <span className="bg-first text-white">
          expense-sorted@txclassify.iam.gserviceaccount.com
        </span>{" "}
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
