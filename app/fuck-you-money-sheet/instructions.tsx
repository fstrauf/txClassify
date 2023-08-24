import Image from "next/image";
import GetItHereButton from "./getItHereButton";
import Link from "next/link";

export default function Instructions() {
  return (
    <section id="instructions">
      <h2>Instructions</h2>
      <p className="prose prose-invert">
        The sheet below is all you need to get tracking.
      </p>
      <GetItHereButton/>
      <p className="prose prose-invert">
        Hop in and make a copy, ideally to a Google Drive.
      </p>
      <div className="mt-6">
        <Image
          width={760 / 2}
          height={546 / 2}
          src="/f-you-make-a-copy.png"
          className="rounded-md"
          alt="Copy the template sheet"
        />
      </div>
      <p className="prose prose-invert">Every month then do the following:</p>
      <ol className="list-decimal ml-8 prose prose-invert mx-auto">
        <li>
          Add your income from different sources to the Income Tab of the
          financial-overview sheet
          <div className="mt-6">
            <Image
              width={752 / 2}
              height={846 / 2}
              src="/f-you-income.png"
              className="rounded-md"
              alt="Add your income to the sheet"
            />
          </div>
        </li>

        <li>
          Add your new expenses from your bank account to the new_dump sheet
          <div className="mt-6">
            <Image
              width={1572 / 2}
              height={846 / 2}
              src="/f-you-new-dump.png"
              className="rounded-md"
              alt="Add new expenses"
            />
          </div>
        </li>
        <li>
          Categorise your expenses{" "}
          <Link href="/demo">head here to use the AI-Tool</Link>
        </li>
        <li>
          Review and adjust categories{" "}
          <div className="mt-6">
            <Image
              width={2174 / 2}
              height={846 / 2}
              src="/f-you-expense-detail.png"
              className="rounded-md"
              alt="Adjust your expense categories"
            />
          </div>
        </li>
        <li>
          Copy over to a new month from the expense overview tab{" "}
          <div className="mt-6">
            <Image
              width={1336 / 2}
              height={846 / 2}
              src="/f-you-expense-overview.png"
              className="rounded-md"
              alt="Calculate your monthly results"
            />
          </div>
        </li>
        <li>
          Review spending, saving, and run-rate{" "}
          <div className="mt-6">
            <Image
              width={1200 / 2}
              height={1076 / 2}
              src="/f-you-stats.png"
              className="rounded-md"
              alt="Calculate your monthly results"
            />
          </div>
        </li>
        <li>Start saving up fuck you money</li>
      </ol>
      <p className="prose prose-invert">
        To get into the right mood of why to use this, I recommend reading this:{" "}
        <a href="https://florianstrauf.substack.com/p/fuck-you-money-doesnt-mean-you-need">
          Fuck You Money Doesn't Mean You Need To Be Rich
        </a>
      </p>
    </section>
  );
}
