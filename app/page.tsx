import Image from "next/image";
import Testimonials from "@/components/Testimonials";
import References from "@/components/References";
import FAQ from "@/components/FAQ";
import Features from "@/components/Features";
import Link from "next/link";
import Head from "next/head";
import { FaGoogle } from "react-icons/fa";

export default function Home() {
  return (
    <div className="min-h-screen bg-background-default">
      <div className="container">
        <Head>
          <link rel="canonical" href="https://www.expensesorted.com/" />
        </Head>
      </div>

      <main className="container mx-auto px-4 py-16 max-w-5xl">
        {/* Hero Section - Emotional heading with clear promise */}
        <div className="text-center mb-12">
          <h1 className="text-5xl font-bold text-gray-900 mb-6 leading-tight">
            Unlock Your{" "}
            <span className="bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
              Financial Freedom
            </span>
            ,
            <br />
            One Expense at a Time
          </h1>
          <div className="flex flex-col gap-4 mb-8 max-w-3xl mx-auto">
            <p className="text-xl text-gray-800">
              Gain complete clarity on your spending, savings, and financial runway
            </p>
            <div className="text-lg text-gray-600 flex flex-col gap-2">
              <p>✨ Automatic expense categorization</p>
              <p>📊 Simple, powerful spreadsheet</p>
              <p>🎯 Turn confusion into confidence</p>
            </div>
          </div>

          {/* Single, Clear CTA */}
          <Link
            href="/fuck-you-money-sheet"
            className="inline-flex items-center px-8 py-4 rounded-xl bg-primary text-white font-semibold hover:bg-primary-dark transition-all duration-200 shadow-soft hover:shadow-glow text-lg"
          >
            Get Your Fuck You Money Spreadsheet
            <svg className="ml-2 w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </Link>

          {/* Quick Social Proof */}
          <div className="mt-8 text-gray-600">
            <p>Over 1,000 people are already on their path to financial independence</p>
          </div>

          <div className="mt-6">
            <Link
              href="https://workspace.google.com/marketplace/app/expense_sorted/456363921097"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center px-6 py-3 rounded-lg bg-white border-2 border-primary text-primary font-semibold hover:bg-primary/10 transition-all duration-200 shadow-sm gap-2"
            >
              <FaGoogle className="w-5 h-5" />
              Get Google Workspace Add-on
            </Link>
            <p className="mt-2 text-sm text-gray-600">Already using Google Sheets? Install directly from Marketplace</p>
          </div>
        </div>

        {/* Product Demo Video */}
        <div className="rounded-2xl overflow-hidden shadow-soft mb-16 bg-surface p-6">
          <div className="aspect-video relative">
            <iframe
              src="https://www.youtube.com/embed/eaCIEWA44Fk"
              title="ExpenseSorted Demo Video"
              allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture"
              allowFullScreen
              className="absolute top-0 left-0 w-full h-full rounded-xl"
            />
          </div>
        </div>

        {/* Core Benefits Section */}
        <div className="bg-surface rounded-2xl p-8 shadow-soft mb-16">
          <h2 className="text-3xl font-bold text-gray-900 mb-8 text-center">
            Why Use the "Fuck You Money Spreadsheet?"
          </h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="text-center p-6">
              <h3 className="text-xl font-semibold mb-4">Instant Clarity</h3>
              <p className="text-gray-600">Clearly see your expenses, savings, and how long you can afford freedom</p>
            </div>
            <div className="text-center p-6">
              <h3 className="text-xl font-semibold mb-4">Save Valuable Time</h3>
              <p className="text-gray-600">Automatic expense categorization replaces tedious manual tracking</p>
            </div>
            <div className="text-center p-6">
              <h3 className="text-xl font-semibold mb-4">Empowering Insights</h3>
              <p className="text-gray-600">Make smarter financial decisions with transparent data at your fingertips</p>
            </div>
          </div>
        </div>

        {/* Founder Message */}
        <div className="bg-surface rounded-2xl p-8 shadow-soft mb-16">
          <div className="max-w-3xl mx-auto">
            <h2 className="text-2xl font-bold text-gray-900 mb-6">A Message from the Founder</h2>
            <p className="text-lg text-gray-600 leading-relaxed">
              "I built this tool after realizing financial freedom isn't about becoming rich—it's about buying back your
              time. With clear expense tracking and the simplicity of knowing your financial runway, you can confidently
              pursue the life you truly want."
            </p>
          </div>
        </div>

        {/* Testimonials Section - Simplified with key quotes */}
        <div className="bg-gradient-to-br from-primary/5 to-secondary/5 rounded-2xl p-8 mb-16">
          <h2 className="text-3xl font-bold text-gray-900 mb-12 text-center">What Our Community Says</h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="bg-white p-6 rounded-xl shadow-soft">
              <p className="text-gray-600 mb-4">"It changed how I look at my finances forever."</p>
              <p className="font-semibold">— Alex M.</p>
            </div>
            <div className="bg-white p-6 rounded-xl shadow-soft">
              <p className="text-gray-600 mb-4">
                "I now know exactly how long my savings will last, giving me peace of mind."
              </p>
              <p className="font-semibold">— Sarah K.</p>
            </div>
            <div className="bg-white p-6 rounded-xl shadow-soft">
              <p className="text-gray-600 mb-4">"No more guessing. Financial freedom feels achievable now."</p>
              <p className="font-semibold">— Liam R.</p>
            </div>
          </div>
        </div>

        {/* FAQ Section */}
        <div className="bg-surface rounded-2xl p-8 shadow-soft mb-16">
          <h2 className="text-3xl font-bold text-gray-900 mb-12 text-center">Common Questions</h2>
          <FAQ />
        </div>

        {/* References Section */}
        <div className="bg-gradient-to-br from-accent/5 to-primary/5 rounded-2xl p-8">
          <References />
        </div>
      </main>
    </div>
  );
}
