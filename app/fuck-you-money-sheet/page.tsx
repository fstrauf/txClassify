import React from "react";
import Image from "next/image";
import Link from "next/link";
import GetItHereButton from "./getItHereButton";
import Instructions from "./instructions";
import Intro from "./intro";
import StayUpToDate from "./stayUpToDate";
import PageHeader from "@/app/components/PageHeader";

export default function FinancialFreedomSheet() {
  return (
    <div className="min-h-screen bg-background-default">
      <main className="container mx-auto px-4 py-8 md:py-16 max-w-7xl">
        <div className="bg-surface rounded-2xl shadow-soft p-4 md:p-8 space-y-6 md:space-y-8">
          <PageHeader title="Financial Freedom - Cost of Living Tracking" />

          {/* Key Benefits Summary - Added for immediate value clarity */}
          <div className="max-w-3xl mx-auto bg-primary/5 rounded-xl p-4 md:p-6 mb-6 md:mb-8">
            <h2 className="text-xl md:text-2xl font-bold text-gray-900 mb-3 md:mb-4">What You'll Get:</h2>
            <ul className="space-y-2 md:space-y-3 text-sm md:text-base">
              <li className="flex items-start">
                <span className="text-primary font-bold mr-2">✓</span>
                <span>Track all your expenses automatically with smart categorization</span>
              </li>
              <li className="flex items-start">
                <span className="text-primary font-bold mr-2">✓</span>
                <span>Calculate your financial runway - exactly how long your savings will last</span>
              </li>
              <li className="flex items-start">
                <span className="text-primary font-bold mr-2">✓</span>
                <span>Visualize your progress toward financial independence</span>
              </li>
              <li className="flex items-start">
                <span className="text-primary font-bold mr-2">✓</span>
                <span>One-time setup, lifetime value - no ongoing subscription fees</span>
              </li>
            </ul>
          </div>

          <div className="max-w-3xl mx-auto">
            <GetItHereButton />

            {/* Featured Testimonial - Added for social proof */}
            <div className="my-6 md:my-8 bg-white p-4 md:p-6 rounded-xl shadow-soft border border-gray-100">
              <div className="flex items-center mb-3 md:mb-4">
                <div className="w-10 h-10 md:w-12 md:h-12 rounded-full bg-gray-200 mr-3 md:mr-4 overflow-hidden">
                  <Image
                    src="/testimonial-featured.jpg"
                    width={48}
                    height={48}
                    alt="Jessie T."
                    className="w-full h-full object-cover"
                  />
                </div>
                <div>
                  <p className="font-semibold">Jessie T.</p>
                  <p className="text-xs md:text-sm text-gray-600">Software Engineer</p>
                </div>
              </div>
              <p className="text-sm md:text-base text-gray-700 italic">
                "This spreadsheet changed my relationship with money. I finally know exactly how long my savings will
                last, which gave me the confidence to negotiate a better salary at work. Worth every penny!"
              </p>
              <div className="flex mt-2">
                <span className="text-yellow-400">★★★★★</span>
              </div>
            </div>

            <StayUpToDate />
            <div className="mt-8 md:mt-12 rounded-xl overflow-hidden shadow-soft hover:shadow-glow transition-all duration-300">
              <ImageComponent />
            </div>
            <div className="mt-6 md:mt-8">
              <Navigation />
            </div>
            <div className="prose prose-lg max-w-none mt-12 prose-headings:text-gray-900 prose-p:text-gray-700 prose-strong:text-gray-900 prose-a:text-primary hover:prose-a:text-primary-dark">
              <Intro />
              <Instructions />
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

function ImageComponent() {
  return (
    <Image
      width={1306}
      height={1230}
      src="/f-you-money-expense-vs-savings.png"
      className="w-full"
      alt="Financial Freedom Spreadsheet"
      priority={true}
    />
  );
}

function Navigation() {
  return (
    <nav className="flex flex-wrap gap-4 justify-center">
      <Link
        href="#intro"
        className="px-6 py-3 rounded-xl bg-primary text-white font-semibold hover:bg-primary-dark transition-all duration-200 shadow-soft hover:shadow-glow"
      >
        Intro
      </Link>
      <Link
        href="#instructions"
        className="px-6 py-3 rounded-xl bg-white text-primary border border-primary/10 font-semibold hover:bg-gray-50 transition-all duration-200 shadow-soft"
      >
        Instructions
      </Link>
    </nav>
  );
}
