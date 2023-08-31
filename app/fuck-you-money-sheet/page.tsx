import React from "react";
import Image from "next/image";
import Link from "next/link";
import GetItHereButton from "./getItHereButton";
import Instructions from "./instructions";
import Intro from "./intro";

export default function FuckYouMoneySheet() {
  return (
    <div className="flex flex-col min-h-screen bg-gradient-to-br from-first via-second to-third">
      <main className="flex-grow flex items-center justify-center p-10">
        <div className="mx-auto w-full prose prose-invert max-w-4xl bg-third p-6 rounded-xl shadow-lg space-y-6">
          <h1 className="text-3xl font-bold leading-tight text-center">
            Fuck you Money - Cost of Living Tracking Google Sheet
          </h1>
          <GetItHereButton />
          <ImageComponent />
          <Navigation />
          <Intro />
          <Instructions />
        </div>
      </main>
    </div>
  );
}

function ImageComponent() {
  return (
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
  );
}

function Navigation() {
  return (
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
      <Link
        className="bg-second hover:bg-third py-2 px-4 rounded-full text-white font-semibold transition duration-300 ease-in-out no-underline"
        href="/demo"
      >
        Try the Demo
      </Link>
    </nav>
  );
}
