import Link from 'next/link';

export default function SimpleGoogleSheetsTracker() {
  return (
    <div className="flex flex-col min-h-screen bg-gradient-to-br from-first via-second to-third">

      <main className="flex-grow flex items-center justify-center p-10">
        <div className="w-full max-w-4xl bg-third p-6 rounded-xl shadow-lg text-white space-y-6">
          <h1 className="text-3xl font-bold leading-tight text-center">
            The Power of Simplicity: Expense Tracker for Google Sheets
          </h1>
          <p className="text-lg text-center">
            In today's complex financial world, there's beauty in simplicity. That's where our simple expense tracker for Google Sheets shines. It strips away the complications and gives you straightforward, effective expense tracking.
          </p>
          <p className="text-lg text-center">
            No bells and whistles, just a clean, user-friendly interface on a platform you're already familiar with. Google Sheets' flexibility combined with our tool's simplicity creates the ultimate budgeting solution.
          </p>
          <h2 className="text-2xl text-first text-center mt-4">
            Embrace Minimalism in Financial Management
          </h2>
          <p className="text-lg text-center">
            Merging innovation with minimalism, our tool is designed for those who value simplicity. Integrate it with Google Sheets, and you have a streamlined expense tracker at your fingertips. Keep track of your finances without the fuss.
          </p>
          <div className="mt-6 text-center">
            <Link href="/" className="bg-first hover:bg-second py-2 px-6 rounded-full text-white font-semibold transition duration-300 ease-in-out">
                Discover Simplicity
            </Link>
          </div>
        </div>
      </main>

    </div>
  );
}
