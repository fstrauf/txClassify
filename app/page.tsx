import Image from "next/image";
import Testimonials from "@/components/Testimonials";
import References from "@/components/References";
import FAQ from "@/components/FAQ";
import Features from "@/components/Features";
import PageHeader from "@/app/components/PageHeader";
import Link from "next/link";
import Head from "next/head";
import Script from "next/script";

export default function Home() {
  return (
    <div className="min-h-screen bg-background-default">
      <div className="container">
        <Script src="https://www.googletagmanager.com/gtag/js?id=G-NB5F14FKZT" />
        <Script id="google-analytics">
          {`
          window.dataLayer = window.dataLayer || [];
          function gtag(){dataLayer.push(arguments);}
          gtag('js', new Date());
 
          gtag('config', 'G-NB5F14FKZT');
        `}
        </Script>
      </div>
      <Head>
        <link rel="canonical" href="https://www.expensesorted.com/" />
      </Head>
      
      <main className="container mx-auto px-4 py-16 max-w-7xl">
        {/* Hero Section */}
        <PageHeader
          title="Smart Expense Categorization"
          subtitle="Stop manually categorizing your expenses. Let AI handle the chores and focus on what matters."
          className="mb-16"
        />

        {/* CTA Buttons */}
        <div className="flex flex-col sm:flex-row gap-4 justify-center mb-12">
          <Link
            href="/fuck-you-money-sheet"
            className="inline-flex items-center px-8 py-4 rounded-xl bg-primary text-white font-semibold hover:bg-primary-dark transition-all duration-200 shadow-soft hover:shadow-glow"
          >
            Get Started with Google Sheets™
            <svg className="ml-2 w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </Link>
          <Link
            href="/api-key"
            className="inline-flex items-center px-8 py-4 rounded-xl bg-white text-primary border border-primary/10 font-semibold hover:bg-gray-50 transition-all duration-200 shadow-soft"
          >
            Get Your API Key
          </Link>
        </div>

        {/* Telegram Group */}
        <div className="bg-surface rounded-xl p-6 shadow-soft max-w-2xl mx-auto mb-16">
          <p className="text-gray-700 mb-3">Join our community of users:</p>
          <a
            href="https://t.me/f_you_money"
            target="_blank"
            rel="noopener noreferrer"
            className="inline-flex items-center text-primary hover:text-primary-dark font-semibold transition-colors"
          >
            <svg className="w-6 h-6 mr-2" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 0C5.373 0 0 5.373 0 12s5.373 12 12 12 12-5.373 12-12S18.627 0 12 0zm5.562 8.161c-.18 1.897-.962 6.502-1.359 8.627-.168.9-.5 1.201-.82 1.23-.697.064-1.226-.461-1.901-.903-1.056-.692-1.653-1.123-2.678-1.799-1.185-.781-.417-1.21.258-1.911.177-.184 3.247-2.977 3.307-3.23.007-.032.014-.15-.056-.212s-.174-.041-.249-.024c-.106.024-1.793 1.139-5.062 3.345-.479.329-.913.489-1.302.481-.428-.008-1.252-.241-1.865-.44-.752-.244-1.349-.374-1.297-.789.027-.216.325-.437.893-.663 3.498-1.524 5.831-2.529 6.998-3.014 3.332-1.386 4.025-1.627 4.476-1.635.099-.002.321.023.465.141.145.118.181.344.203.483.023.139.041.562.041.562z"/>
            </svg>
            @f_you_money on Telegram
          </a>
        </div>

        {/* Demo Image */}
        <div className="rounded-2xl overflow-hidden shadow-soft hover:shadow-glow transition-shadow duration-300">
          <Image
            width={1306}
            height={1229}
            src="/f-you-money-expense-detail.png"
            className="w-full shadow-lg"
            alt="Expense Sorted Demo"
          />
        </div>

        {/* Features Section */}
        <div className="bg-surface rounded-2xl p-8 shadow-soft mb-16">
          <h2 className="text-3xl font-bold text-gray-900 mb-12 text-center pb-2 leading-normal">
            Why Choose Our Solution?
          </h2>
          <Features />
        </div>

        {/* Testimonials Section */}
        <div className="bg-gradient-to-br from-primary/5 to-secondary/5 rounded-2xl p-8 mb-16">
          <h2 className="text-3xl font-bold text-gray-900 mb-12 text-center pb-2 leading-normal">
            What Our Users Say
          </h2>
          <Testimonials />
        </div>

        {/* FAQ Section */}
        <div className="bg-surface rounded-2xl p-8 shadow-soft mb-16">
          <h2 className="text-3xl font-bold text-gray-900 mb-12 text-center pb-2 leading-normal">
            Frequently Asked Questions
          </h2>
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
