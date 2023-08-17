import Link from "next/link";

export default function BankTransactionClassification() {
  return (
    <div className="flex flex-col min-h-screen bg-gradient-to-br from-first via-second to-third">
      <main className="flex-grow flex items-center justify-center p-10">
        <div className="w-full max-w-4xl bg-third p-6 rounded-xl shadow-lg text-white space-y-6">
          <h1 className="text-3xl font-bold leading-tight text-center">
            Mastering Bank Transaction Classification
          </h1>
          <p className="text-lg text-center">
            As the digital financial landscape expands, the need for organized
            bank transactions has never been more paramount. Bank transaction
            classification stands as the linchpin for effective financial
            management and clarity.
          </p>
          <p className="text-lg text-center">
            By classifying bank transactions effectively, individuals and
            businesses can effortlessly track expenses, monitor revenue streams,
            and maintain robust financial health.
          </p>
          <h2 className="text-2xl text-first text-center mt-4">
            Elevate Financial Oversight with Precision
          </h2>
          <p className="text-lg text-center">
            Our solution offers unparalleled bank transaction classification,
            utilizing cutting-edge algorithms and user-friendly interfaces.
            Seamlessly categorize and analyze your banking activities, and lay
            the foundation for proactive financial decisions.
          </p>
          <div className="mt-6 text-center">
            <Link
              href="/"
              className="bg-first hover:bg-second py-2 px-6 rounded-full text-white font-semibold transition duration-300 ease-in-out"
            >
              Delve into Classification
            </Link>
          </div>
        </div>
      </main>
    </div>
  );
}
