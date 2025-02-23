import React from "react";
import Image from "next/image";
import Link from "next/link";
import GetItHereButton from "./getItHereButton";
import Instructions from "./instructions";
import Intro from "./intro";
import StayUpToDate from "./stayUpToDate";
import PageHeader from "@/app/components/PageHeader";

export default function FuckYouMoneySheet() {
  return (
    <div className="min-h-screen bg-background-default">
      <main className="container mx-auto px-4 py-16 max-w-7xl">
        <div className="bg-surface rounded-2xl shadow-soft p-8 space-y-8">
          <PageHeader
            title="Fuck you Money - Cost of Living Tracking"
          />
          <div className="max-w-3xl mx-auto">
            <GetItHereButton />
            <StayUpToDate />
            <div className="mt-12 rounded-xl overflow-hidden shadow-soft hover:shadow-glow transition-all duration-300">
              <ImageComponent />
            </div>
            <div className="mt-8">
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
      width={3214 / 2}
      height={1356 / 2}
      src="/f-you-money-overview.png"
      className="w-full"
      alt="Expense Sorter"
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
