import Link from "next/link";

export default function References() {
  return (
    <div>
      <div className="mt-8 flex flex-wrap justify-around items-center space-y-4">
        <div className="bg-white rounded-xl p-6 shadow-soft flex flex-col items-center">
          <h3 className="text-2xl font-semibold text-gray-900 mb-4">Use Cases:</h3>
          <div className="flex flex-col items-center space-y-2">
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
              Google Sheets™ Expense Tracking
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
}
