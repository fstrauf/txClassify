import Image from "next/image";
import Testimonials from "../components/Testimonials";
import References from "../components/References";
import FAQ from "../components/FAQ";
import Features from "../components/Features";
import Link from "next/link";
import Head from "next/head";
import Script from "next/script";

export default function Home() {
  return (
    <div className="flex flex-col min-h-screen bg-gradient-to-br from-first via-second to-third">
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
      <main className="flex-grow flex items-center justify-center p-10">
        <div className="w-full max-w-4xl bg-third p-6 rounded-xl shadow-lg text-white space-y-6">
          <h1 className="text-3xl font-bold leading-tight text-center">
            Avoid the chores of manually categorising your expenses every month.
          </h1>
          <h2 className="text-2xl text-first text-center">Use AI instead!</h2>
          <p className="text-lg text-center">
            Hook this up to your Google Sheet and speed up your monthly
            workflow.
          </p>
          <div className="text-center">
            <Link
              className="bg-first hover:bg-second py-2 px-6 rounded-full text-white font-semibold transition duration-300 ease-in-out inline-block"
              href="/fuck-you-money-sheet"
            >
              Get the Google Sheet Template
            </Link>
          </div>
          <div className="text-center">
            <Link
              href="/demo"
              className="bg-first hover:bg-second py-2 px-6 rounded-full text-white font-semibold transition duration-300 ease-in-out"
            >
              Try it out now
            </Link>
          </div>
          <div className="mt-6">
            <Image
              width={852}
              height={762}
              src="/expense-sorter-main.png"
              className="rounded-md"
              alt="Expense Sorter"
            />
          </div>
          <h1 className="pt-20 text-3xl font-bold leading-tight text-center">
            You want to learn more you said?
          </h1>
          <div className="flex flex-col gap-20">
            <Features />
            <Testimonials />
            <FAQ />
            <References />
          </div>
        </div>
      </main>
    </div>
  );
}
