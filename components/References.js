import Link from "next/link";

export default function References() {
  return (
    <div>
      <div className="mt-8 text-center">
        <Link
          href="/demo"
          className="bg-first hover:bg-second py-2 px-6 rounded-full text-white font-semibold transition duration-300 ease-in-out"
        >
          Try a Demo
        </Link>
      </div>
      <div className="mt-8 flex flex-wrap justify-around items-center space-y-4">
        <div className="flex flex-col items-center space-x-4 mt-4">
          <h3 className="text-xl font-semibold pr-2">Use Cases:</h3>
          <Link
            href="/use-cases/bank-transaction-classification"
            className="px-4 py-1 text-first hover:bg-third rounded hover:underline"
          >
            Bank Transactions
          </Link>
          <Link
            href="/use-cases/expense-classification"
            className="px-4 py-1 text-first hover:bg-third rounded hover:underline"
          >
            Expense Classification
          </Link>
          <Link
            href="/use-cases/expense-tracker-google-sheet"
            className="px-4 py-1 text-first hover:bg-third rounded hover:underline"
          >
            Expense Tracking
          </Link>
          <Link
            href="/use-cases/simple-expense-tracker-google-sheet"
            className="px-4 py-1 text-first hover:bg-third rounded hover:underline"
          >
            Google Sheet Expense Tracking
          </Link>
        </div>
      </div>
    </div>
  );
}
