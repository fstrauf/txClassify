import Link from 'next/link';

export default function GoogleSheetsTracker() {
  return (
    <div className="flex flex-col min-h-screen bg-gradient-to-br from-first via-second to-third">

      <main className="flex-grow flex items-center justify-center p-10">
        <div className="w-full max-w-4xl bg-third p-6 rounded-xl shadow-lg text-white space-y-6">
          <h1 className="text-3xl font-bold leading-tight text-center">
            Revolutionize Budgeting with an Expense Tracker for Google Sheets
          </h1>
          <p className="text-lg text-center">
            Keeping track of expenses is no longer a tedious task confined to traditional software. The fusion of expense tracking with the versatility of Google Sheets offers a dynamic solution for your financial management needs.
          </p>
          <p className="text-lg text-center">
            Benefit from the collaboration, accessibility, and customization that Google Sheets provides. Our tool seamlessly bridges the gap, ensuring you have a top-notch expense tracking experience.
          </p>
          <h2 className="text-2xl text-first text-center mt-4">
            Integrating Technology with Simplicity
          </h2>
          <p className="text-lg text-center">
            Dive into the future of personal finance management. Our tool integrates effortlessly with Google Sheets, turning it into a powerful expense tracker. No software downloads, no new platforms to learn—just your familiar Sheets interface enhanced with our technology.
          </p>
          <div className="mt-6 text-center">
            <Link href="/" className="bg-first hover:bg-second py-2 px-6 rounded-full text-white font-semibold transition duration-300 ease-in-out">
                Explore the Tracker
            </Link>
          </div>
        </div>
      </main>

    </div>
  );
}
