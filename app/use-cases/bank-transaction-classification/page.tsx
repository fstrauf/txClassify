import Link from "next/link";

export default function BankTransactionClassification() {
  return (
    <div className="min-h-screen bg-background-default">
      <main className="container mx-auto px-4 py-16 max-w-7xl">
        <div className="bg-surface rounded-2xl shadow-soft p-8 space-y-8">
          <h1 className="text-4xl md:text-5xl font-bold text-gray-900 text-center bg-clip-text text-transparent bg-gradient-to-r from-primary-dark via-primary to-secondary animate-gradient">
            Mastering Bank Transaction Classification
          </h1>
          <div className="max-w-3xl mx-auto space-y-8">
            <p className="text-xl text-gray-700 text-center">
              As the digital financial landscape expands, the need for organized
              bank transactions has never been more paramount. Bank transaction
              classification stands as the linchpin for effective financial
              management and clarity.
            </p>
            <p className="text-xl text-gray-700 text-center">
              By classifying bank transactions effectively, individuals and
              businesses can effortlessly track expenses, monitor revenue streams,
              and maintain robust financial health.
            </p>
            <div className="pt-8">
              <h2 className="text-3xl font-bold text-gray-900 text-center mb-6">
                Elevate Financial Oversight with Precision
              </h2>
              <p className="text-xl text-gray-700 text-center">
                Our solution offers unparalleled bank transaction classification,
                utilizing cutting-edge algorithms and user-friendly interfaces.
                Seamlessly categorize and analyze your banking activities, and lay
                the foundation for proactive financial decisions.
              </p>
            </div>
            <div className="text-center pt-4">
              <Link
                href="/"
                className="px-6 py-3 rounded-xl bg-primary text-white font-semibold hover:bg-primary-dark transition-all duration-200 shadow-soft hover:shadow-glow inline-flex items-center"
              >
                Delve into Classification
                <svg className="ml-2 w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                </svg>
              </Link>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
