import Link from 'next/link';

export default function ExpenseClassification() {
  return (
    <div className="min-h-screen bg-background-default">
      <main className="container mx-auto px-4 py-16 max-w-7xl">
        <div className="bg-surface rounded-2xl shadow-soft p-8 space-y-8">
          <h1 className="text-4xl md:text-5xl font-bold text-gray-900 text-center bg-clip-text text-transparent bg-gradient-to-r from-primary-dark via-primary to-secondary animate-gradient">
            The Significance of Expense Classification
          </h1>
          <div className="max-w-3xl mx-auto space-y-8">
            <p className="text-xl text-gray-700 text-center">
              Expense classification isn't just about keeping your bank transactions tidy. 
              It's a crucial aspect of financial planning, budgeting, and effective money management.
            </p>
            <p className="text-xl text-gray-700 text-center">
              Through accurate classification, you can identify spending patterns, 
              make informed financial decisions, and stay on top of your financial goals.
            </p>
            <div className="pt-8">
              <h2 className="text-3xl font-bold text-gray-900 text-center mb-6">
                Simplify with AI-powered Classification
              </h2>
              <p className="text-xl text-gray-700 text-center">
                Using our AI-driven tool, you can seamlessly integrate expense classification 
                into your daily routine. No more manual categorization. Just link it to your 
                Google Sheets™, and let the technology handle the rest.
              </p>
            </div>
            <div className="text-center pt-4">
              <Link
                href="/"
                className="px-6 py-3 rounded-xl bg-primary text-white font-semibold hover:bg-primary-dark transition-all duration-200 shadow-soft hover:shadow-glow inline-flex items-center"
              >
                Explore the Tool
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
