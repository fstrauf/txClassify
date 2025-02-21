import Link from 'next/link';

export default function ExpenseClassification() {
  return (
    <div className="flex flex-col min-h-screen bg-gradient-to-br from-first via-second to-third">


      <main className="flex-grow flex items-center justify-center p-10">
        <div className="w-full max-w-4xl bg-third p-6 rounded-xl shadow-lg text-white space-y-6">
          <h1 className="text-3xl font-bold leading-tight text-center">
            The Significance of Expense Classification
          </h1>
          <p className="text-lg text-center">
            Expense classification isn't just about keeping your bank transactions tidy. It's a crucial aspect of financial planning, budgeting, and effective money management.
          </p>
          <p className="text-lg text-center">
            Through accurate classification, you can identify spending patterns, make informed financial decisions, and stay on top of your financial goals.
          </p>
          <h2 className="text-2xl text-first text-center mt-4">
            Simplify with AI-powered Classification
          </h2>
          <p className="text-lg text-center">
            Using our AI-driven tool, you can seamlessly integrate expense classification into your daily routine. No more manual categorization. Just link it to your Google Sheets™, and let the technology handle the rest.
          </p>
          <div className="mt-6 text-center">
            <Link href="/" className="bg-first hover:bg-second py-2 px-6 rounded-full text-white font-semibold transition duration-300 ease-in-out">
                Explore the Tool
            </Link>
          </div>
        </div>
      </main>
    </div>
  );
}
