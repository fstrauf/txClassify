import React from "react";
import Image from "next/image";
import Link from "next/link";

export default function FuckYouMoneySheet() {
  const getItHereButton = () => (
    <div className="text-center">
      <a
        className="bg-first hover:bg-second py-2 px-6 rounded-full text-white font-semibold transition duration-300 ease-in-out inline-block no-underline"
        href="https://docs.google.com/spreadsheets/d/1Buon6FEg7JGJMjuZgNgIrm5XyfP38JeaOJTNv6YQSHA/edit#gid=1128667954"
      >
        Get It Here
      </a>
    </div>
  );

  return (
    <div className="flex flex-col min-h-screen bg-gradient-to-br from-first via-second to-third">
      <main className="flex-grow flex items-center justify-center p-10">
        <div className="mx-auto w-full prose prose-invert max-w-4xl bg-third p-6 rounded-xl shadow-lg space-y-6">
          <h1 className="text-3xl font-bold leading-tight text-center">
            Fuck you Money - Cost of Living Tracking Google Sheet
          </h1>
          {getItHereButton()}
          <div className="mt-6">
            <Image
              width={3214 / 2}
              height={1356 / 2}
              src="/f-you-money-overview.png"
              className="rounded-md"
              alt="Expense Sorter"
              priority={true}
            />
          </div>
          <nav className="flex space-x-4">
            <Link
              className="bg-second hover:bg-third py-2 px-4 rounded-full text-white font-semibold transition duration-300 ease-in-out no-underline"
              href="#intro"
            >
              Intro
            </Link>
            <Link
              className="bg-second hover:bg-third py-2 px-4 rounded-full text-white font-semibold transition duration-300 ease-in-out no-underline"
              href="#instructions"
            >
              Instructions
            </Link>
          </nav>
          <section id="intro">
            <h2>Intro</h2>
            <p className="prose prose-invert">
              Introducing the Fuck You Money Spreadsheet: Take Control of Your
              Financial Independence! Discover the power of Fuck You Money—a
              term that signifies the freedom to say "fuck you" to unfavorable
              situations or people. Achieving financial independence reduces
              stress, provides security, and grants the freedom to choose your
              path. But you don't have to be rich or retire forever to
              experience it.
            </p>
            <p className="prose prose-invert">
              Think small and focus on buying yourself time—time to find a new
              job, start a business, or embark on exciting adventures. The Fuck
              You Money Spreadsheet helps you understand your financial
              parameters and calculate your runway—the time you can afford to
              take off based on your savings and expenses. By tracking your
              expenses and savings, you can uncover the stored time in your
              account. Simply subtract your expenses from your savings to
              determine how long you can sustain yourself without a job. It's a
              straightforward formula that empowers you to make informed
              decisions about your finances and prioritize what truly matters.
            </p>
            <p className="prose prose-invert">
              The Fuck You Money Spreadsheet encourages you to lower your
              expenses and increase your income, offering a simple yet effective
              way to accumulate more time in your account. It aligns with the
              principles of the FIRE movement (Financial Independence, Retire
              Early), which emphasizes saving and investing to achieve financial
              freedom. With this spreadsheet, you gain awareness of your
              financial parameters, enabling you to take control and work
              towards increasing your stored time.
            </p>
            <p className="prose prose-invert">
              Having a cushion of fuck you money provides peace of mind and
              opens doors to new opportunities. Take charge of your financial
              independence today. Get the Fuck You Money Spreadsheet and unlock
              the potential of your time. Embrace a life with fewer constraints
              and more freedom to pursue your passions.
            </p>
          </section>
          <section id="instructions">
            <h2>Instructions</h2>
            <p className="prose prose-invert">
              The sheet below is all you need to get tracking.
            </p>
            {getItHereButton()}
            <p className="prose prose-invert">
              Hop in and make a copy, ideally to a Google Drive.
            </p>
            <div className="mt-6">
              <Image
                width={760 / 2}
                height={546 / 2}
                src="/f-you-make-a-copy.png"
                className="rounded-md"
                alt="Copy the template sheet"
              />
            </div>
            <p className="prose prose-invert">
              Every month then do the following:
            </p>
            <ol className="list-decimal ml-8 prose prose-invert mx-auto">
              <li>
                Add your income from different sources to the Income Tab of the
                financial-overview sheet
                <div className="mt-6">
                  <Image
                    width={752 / 2}
                    height={846 / 2}
                    src="/f-you-income.png"
                    className="rounded-md"
                    alt="Add your income to the sheet"
                  />
                </div>
              </li>

              <li>
                Add your new expenses from your bank account to the new_dump
                sheet
                <div className="mt-6">
                  <Image
                    width={1572 / 2}
                    height={846 / 2}
                    src="/f-you-new-dump.png"
                    className="rounded-md"
                    alt="Add new expenses"
                  />
                </div>
              </li>
              <li>
                Categorise your expenses{" "}
                <Link href="/demo">head here to use the AI-Tool</Link>
              </li>
              <li>
                Review and adjust categories{" "}
                <div className="mt-6">
                  <Image
                    width={2174 / 2}
                    height={846 / 2}
                    src="/f-you-expense-detail.png"
                    className="rounded-md"
                    alt="Adjust your expense categories"
                  />
                </div>
              </li>
              <li>
                Copy over to a new month from the expense overview tab{" "}
                <div className="mt-6">
                  <Image
                    width={1336 / 2}
                    height={846 / 2}
                    src="/f-you-expense-overview.png"
                    className="rounded-md"
                    alt="Calculate your monthly results"
                  />
                </div>
              </li>
              <li>
                Review spending, saving, and run-rate{" "}
                <div className="mt-6">
                  <Image
                    width={1200 / 2}
                    height={1076 / 2}
                    src="/f-you-stats.png"
                    className="rounded-md"
                    alt="Calculate your monthly results"
                  />
                </div>
              </li>
              <li>Start saving up fuck you money</li>
            </ol>
            <p className="prose prose-invert">
              To get into the right mood of why to use this, I recommend reading
              this:{" "}
              <a href="https://florianstrauf.substack.com/p/fuck-you-money-doesnt-mean-you-need">
                Fuck You Money Doesn't Mean You Need To Be Rich
              </a>
            </p>
          </section>
        </div>
      </main>
    </div>
  );
}
