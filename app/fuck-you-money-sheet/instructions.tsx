import Image from "next/image";
import GetItHereButton from "./getItHereButton";

export default function Instructions() {
  return (
    <section id="instructions">
      <h2>Instructions</h2>
      <p className="text-gray-700">The sheet below is all you need to get tracking.</p>
      <GetItHereButton />
      <p className="text-gray-700">Hop in and make a copy.</p>
      <div className="mt-6">
        <Image
          width={760 / 2}
          height={546 / 2}
          src="/f-you-make-a-copy.png"
          className="rounded-md shadow-lg"
          alt="Copy the template sheet"
        />
      </div>
      <p className="text-gray-700">
        Next, get the{" "}
        <a
          href="https://workspace.google.com/u/0/marketplace/app/expense_sorted/456363921097?flow_type=2"
          className="text-primary hover:text-primary-dark underline"
          target="_blank"
          rel="noopener noreferrer"
        >
          Expense Sorted extension
        </a>{" "}
        from the Google Sheets™ Extension Marketplace
      </p>
      <div className="mt-6">
        <Image
          width={1306 / 1.5}
          height={1230 / 1.5}
          src="/f-you-money-expense-sorted-extension.png"
          className="rounded-md shadow-lg"
          alt="Copy the template sheet"
        />
      </div>
      <p className="text-gray-700">Every month then do the following:</p>
      <ol className="list-decimal ml-8 text-gray-700 mx-auto space-y-6">
        <li>
          Start off by training the model on your current expenses. That way it will now, how to categorise your future
          expenses.
          <div className="mt-6">
            <Image
              width={1306}
              height={1229}
              src="/f-you-money-expense-detail.png"
              className="rounded-md shadow-lg"
              alt="Add your income to the sheet"
              sizes="(max-width: 768px) 100vw, 768px"
              quality={100}
            />
          </div>
        </li>
        <li>Check the Stats and Log tabs for progress.</li>
        <li>
          Add your new expenses from your bank account to the new_transactions sheet and categorise your expenses via
          the Expense Sorted extension
          <div className="mt-6">
            <Image
              width={929}
              height={327}
              src="/f-you-money-new_transactions.png"
              className="rounded-md shadow-lg"
              alt="Add new expenses"
            />
          </div>
        </li>
        <li>Once complete, copy all transactions over to the Expense-Detail tab and adjust categories as needed.</li>
        <li>
          Copy over a new row in the Monthly Expense tab and fill in the new month.
          <div className="mt-6">
            <Image
              width={865}
              height={372}
              src="/f-you-money-monthly-expenses.png"
              className="rounded-md shadow-lg"
              alt="Copy over a new row in the Monthly Expense tab and fill in the new month."
            />
          </div>
        </li>
        <li>
          Review spending, saving, and run-rate in the Expenses vs. Saving tab.
          <div className="mt-6">
            <Image
              width={1200}
              height={1076}
              src="/f-you-money-expense-vs-savings.png"
              className="rounded-md shadow-lg"
              alt="Review spending, saving, and run-rate in the Expenses vs. Saving tab."
              sizes="(max-width: 768px) 100vw, 768px"
              quality={100}
            />
          </div>
        </li>
        <li>Start building your financial freedom fund</li>
      </ol>
      <p className="text-gray-700 mt-6">
        To get into the right mood of why to use this, I recommend reading this:{" "}
        <a
          href="https://ffstrauf.substack.com/p/fuck-you-money-doesnt-mean-you-need"
          className="text-primary hover:text-primary-dark underline"
        >
          Financial Freedom Doesn't Mean You Need To Be Rich
        </a>
      </p>
    </section>
  );
}
