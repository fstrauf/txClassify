import Link from "next/link";

export default function About() {
  return (
    <div className="flex flex-col min-h-screen bg-gradient-to-br from-first via-second to-third">
      <main className="flex-grow flex items-center justify-center p-10">
        <div className="w-full max-w-4xl bg-third p-6 rounded-xl shadow-lg text-white space-y-6 prose prose-invert">
          <h1 className="text-3xl font-bold leading-tight text-center">
            About Expense Sorted
          </h1>
          <p className="text-lg text-center">
            This is how I do my expenses. I thought the approach was quite good,
            so I built this page to share it with others.
          </p>
          <p>
            I regularly post updates and other content on this topic -{" "}
            <a className="underline" href="https://twitter.com/ffstrauf">
              follow me on Twitter
            </a>{" "}
            or{" "}
            <a className="underline" href="https://florianstrauf.substack.com/">
              Substack
            </a>
            .
          </p>
          <h2 className="pt-20 text-3xl font-bold leading-tight text-center">
            Our Mission
          </h2>
          <p className="text-lg text-center">
            Most people currently use something like rule-based categorisation.
            I want to simplify this. I hope that by using the{" "}
            <Link href="/fuck-you-money-sheet" className="underline">
              Google Sheets™ Template
            </Link>{" "}
            in combination with the{" "}
            <Link href="/demo" className="underline">
              categorisation tool
            </Link>{" "}
            you can improve your monthly workflow.
          </p>
        </div>
      </main>
    </div>
  );
}
