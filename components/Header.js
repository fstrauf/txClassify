"use client";
import React, { useState } from "react";
import Image from "next/image";
import Link from "next/link";
import { NavBarButtons } from "../components/nav-bar-buttons";

export default function Header() {
  const [isMenuOpen, setIsMenuOpen] = useState(false);

  return (
    <header className="w-full bg-surface shadow-soft sticky top-0 z-50">
      <div className="container mx-auto px-4 py-3 flex flex-col sm:flex-row items-center justify-between">
        <div className="flex items-center justify-between w-full sm:w-auto">
          <Link href="/" className="flex items-center">
            <Image
              width={220}
              height={40}
              src="/expense_sorted.svg"
              alt="Expense Sorted"
              className="w-[160px] h-auto"
              priority
            />
          </Link>
          <button className="sm:hidden block text-gray-600" onClick={() => setIsMenuOpen(!isMenuOpen)}>
            <svg className="h-6 w-6" viewBox="0 0 20 20" fill="currentColor">
              {isMenuOpen ? (
                <path
                  fillRule="evenodd"
                  d="M4.293 4.293a1 1 0 011.414 0L10 8.586l4.293-4.293a1 1 0 111.414 1.414L11.414 10l4.293 4.293a1 1 0 01-1.414 1.414L10 11.414l-4.293 4.293a1 1 0 01-1.414-1.414L8.586 10 4.293 5.707a1 1 0 010-1.414z"
                  clipRule="evenodd"
                />
              ) : (
                <path
                  fillRule="evenodd"
                  d="M3 5h14a1 1 0 010 2H3a1 1 0 110-2zm0 4h14a1 1 0 010 2H3a1 1 0 010-2zm0 4h14a1 1 0 010 2H3a1 1 0 110-2z"
                  clipRule="evenodd"
                />
              )}
            </svg>
          </button>
        </div>
        <div className="sm:flex hidden">
          <NavBarButtons />
        </div>
        {isMenuOpen && (
          <div className="sm:hidden w-full mt-4">
            <NavBarButtons />
          </div>
        )}
      </div>
    </header>
  );
}
