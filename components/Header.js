// components/Header.js
import Image from "next/image";
import Link from "next/link";

export default function Header() {
  return (
    <header className="w-full bg-[#1E1E1E] text-white p-4 flex items-center">
      <div className="relative">
        <Image
          width={32}
          height={32}
          src="/128_logo_es.png"
          className="rounded-md shadow-xl"
          alt="Token Editor Flow"
        />
      </div>
      <h1 className="ml-3 text-xl">
        <Link href="/">Expense Sorted</Link>
      </h1>
    </header>
  );
}
