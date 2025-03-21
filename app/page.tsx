"use client";
import Image from "next/image";
import Testimonials from "@/components/Testimonials";
import References from "@/components/References";
import FAQ from "@/components/FAQ";
import Features from "@/components/Features";
import Link from "next/link";
import Head from "next/head";
import { FaGoogle } from "react-icons/fa";
import { useState, useEffect } from "react";

export default function Home() {
  const [showAlternateButton, setShowAlternateButton] = useState(false);

  useEffect(() => {
    // Simple A/B testing based on random assignment
    // In production, you'd use a more sophisticated A/B testing framework
    const abTestValue = Math.random() > 0.5;
    setShowAlternateButton(abTestValue);

    // You could also store this in localStorage to maintain consistency for the user
    localStorage.setItem("useAlternateButton", String(abTestValue));
  }, []);

  return (
    <div className="min-h-screen bg-background-default">
      <div className="container">
        <Head>
          <link rel="canonical" href="https://www.expensesorted.com/" />
        </Head>
      </div>

      <main className="container mx-auto px-4 py-8 md:py-16 max-w-5xl">
        {/* Hero Section - Emotional heading with clear promise */}
        <div className="text-center mb-8 md:mb-12">
          <h1 className="text-4xl md:text-5xl font-bold text-gray-900 mb-4 md:mb-6 leading-tight">
            Unlock Your{" "}
            <span className="bg-gradient-to-r from-primary to-secondary bg-clip-text text-transparent">
              Financial Freedom
            </span>
            ,
            <br className="hidden md:block" />
            <span className="md:hidden"> </span>One Expense at a Time
          </h1>
          <div className="flex flex-col gap-3 md:gap-4 mb-6 md:mb-8 max-w-3xl mx-auto">
            <p className="text-lg md:text-xl text-gray-800">
              Track your spending, savings, and financial runway in minutes – without complicated apps
            </p>
            <div className="text-base md:text-lg text-gray-600 flex flex-col gap-2">
              <p>✨ Automatic expense categorization</p>
              <p>📊 Simple, powerful spreadsheet</p>
              <p>🎯 Turn confusion into confidence</p>
            </div>
          </div>

          {/* A/B Testing CTA Buttons */}
          {showAlternateButton ? (
            <Link
              href="/fuck-you-money-sheet"
              className="inline-flex items-center px-6 py-3 md:px-8 md:py-4 rounded-xl bg-primary text-white font-semibold hover:bg-primary-dark transition-all duration-200 shadow-soft hover:shadow-glow text-base md:text-lg w-full md:w-auto justify-center md:justify-start"
            >
              Get Your Financial Freedom Spreadsheet
              <svg className="ml-2 w-4 h-4 md:w-5 md:h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </Link>
          ) : (
            <Link
              href="/fuck-you-money-sheet"
              className="inline-flex items-center px-6 py-3 md:px-8 md:py-4 rounded-xl bg-primary text-white font-semibold hover:bg-primary-dark transition-all duration-200 shadow-soft hover:shadow-glow text-base md:text-lg w-full md:w-auto justify-center md:justify-start"
            >
              Get Your Financial Freedom Spreadsheet
              <svg className="ml-2 w-4 h-4 md:w-5 md:h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </Link>
          )}
          <p className="mt-2 text-sm text-gray-600">
            Track your finances and calculate how long your savings will last
          </p>

          {/* Quick Social Proof */}
          <div className="mt-6 text-gray-600">
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
          <div className="mt-6 text-center">
            <Link
              href="/fuck-you-money-sheet"
              className="inline-flex items-center px-6 py-3 rounded-lg bg-primary text-white font-semibold hover:bg-primary-dark transition-all duration-200 shadow-soft"
            >
              Get Started Now
              <svg className="ml-2 w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </Link>
          </div>
        </div>

        {/* Core Benefits Section */}
        <div className="bg-surface rounded-2xl p-8 shadow-soft mb-16">
          <h2 className="text-3xl font-bold text-gray-900 mb-8 text-center">
            Why Use the "Financial Freedom Spreadsheet?"
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
          <div className="mt-8 text-center">
            <Link
              href="/fuck-you-money-sheet"
              className="inline-flex items-center px-6 py-3 rounded-xl border border-primary text-primary font-semibold hover:bg-primary/5 transition-all duration-200"
            >
              Learn more about the spreadsheet
              <svg className="ml-2 w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </Link>
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
            <div className="mt-6 text-right">
              <Link
                href="/fuck-you-money-sheet"
                className="text-primary hover:underline hover:text-primary-dark transition-all duration-200 font-medium"
              >
                Try it yourself →
              </Link>
            </div>
          </div>
        </div>

        {/* Testimonials Section - Enhanced with faces and more details */}
        <div className="bg-gradient-to-br from-primary/5 to-secondary/5 rounded-2xl p-8 mb-16">
          <h2 className="text-3xl font-bold text-gray-900 mb-12 text-center">What Our Community Says</h2>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="bg-white p-6 rounded-xl shadow-soft flex flex-col items-center">
              <div className="w-16 h-16 rounded-full bg-gray-200 mb-4 overflow-hidden">
                <Image
                  src="/testimonial-alex.jpg"
                  width={64}
                  height={64}
                  alt="Alex M."
                  className="w-full h-full object-cover"
                />
              </div>
              <p className="text-gray-600 mb-4 text-center">
                "It changed how I look at my finances forever. Now I know exactly where my money goes."
              </p>
              <p className="font-semibold">— Alex M.</p>
              <p className="text-sm text-gray-500">Software Engineer</p>
            </div>
            <div className="bg-white p-6 rounded-xl shadow-soft flex flex-col items-center">
              <div className="w-16 h-16 rounded-full bg-gray-200 mb-4 overflow-hidden">
                <Image
                  src="/testimonial-sarah.jpg"
                  width={64}
                  height={64}
                  alt="Sarah K."
                  className="w-full h-full object-cover"
                />
              </div>
              <p className="text-gray-600 mb-4 text-center">
                "I now know exactly how long my savings will last, giving me peace of mind to make career changes."
              </p>
              <p className="font-semibold">— Sarah K.</p>
              <p className="text-sm text-gray-500">Marketing Consultant</p>
            </div>
            <div className="bg-white p-6 rounded-xl shadow-soft flex flex-col items-center">
              <div className="w-16 h-16 rounded-full bg-gray-200 mb-4 overflow-hidden">
                <Image
                  src="/testimonial-liam.jpg"
                  width={64}
                  height={64}
                  alt="Liam R."
                  className="w-full h-full object-cover"
                />
              </div>
              <p className="text-gray-600 mb-4 text-center">
                "No more guessing. Financial freedom feels achievable now that I have a clear roadmap."
              </p>
              <p className="font-semibold">— Liam R.</p>
              <p className="text-sm text-gray-500">Small Business Owner</p>
            </div>
          </div>
        </div>

        {/* FAQ Section */}
        <div className="bg-surface rounded-2xl p-8 shadow-soft mb-16">
          <h2 className="text-3xl font-bold text-gray-900 mb-12 text-center">Common Questions</h2>
          <FAQ />
          <div className="mt-8 text-center">
            <p className="text-gray-600 mb-4">Ready to take control of your financial future?</p>
            <Link
              href="/fuck-you-money-sheet"
              className="inline-flex items-center px-6 py-3 rounded-lg bg-primary text-white font-semibold hover:bg-primary-dark transition-all duration-200 shadow-soft"
            >
              Get Started Now
              <svg className="ml-2 w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
              </svg>
            </Link>
          </div>
        </div>

        {/* References Section */}
        <div className="bg-gradient-to-br from-accent/5 to-primary/5 rounded-2xl p-8">
          <References />
        </div>
      </main>
    </div>
  );
}
