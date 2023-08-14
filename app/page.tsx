import Image from "next/image";

export default function Home() {
  return (
    <div className="flex flex-col min-h-screen bg-gradient-to-br from-first via-second to-third">

      <main className="flex-grow flex items-center justify-center p-10">
        <div className="w-full max-w-4xl bg-third p-6 rounded-xl shadow-lg text-white space-y-6">
          <h1 className="text-3xl font-bold leading-tight text-center">Avoid the chores of manually categorising your expenses every month.</h1>
          <h2 className="text-2xl text-first text-center">Use AI instead!</h2>
          <p className="text-lg text-center">Hook this up to your Google Sheet and speed up your monthly workflow.</p>
          <div className="mt-6">
            <Image
              width={852}
              height={762}
              src="/expense-sorter-main.png"
              className="rounded-md"
              alt="Expense Sorter"
            />
          </div>
        </div>
      </main>
    </div>
  );
}
