// components/Footer.js

import Link from "next/link";

export default function Footer() {
  return (
    <footer className="w-full bg-[#1E1E1E] text-white p-4">
      <div className="flex space-x-4">
        <Link href="/privacy-policy" className="hover:underline text-xs">
          Privacy Policy
        </Link>
        <Link href="/support" className="hover:underline text-xs">
          Support
        </Link>
        <Link href="/terms-of-service" className="hover:underline text-xs">
          Terms of Service
        </Link>
        <Link href="/about" className="hover:underline text-xs">
          About
        </Link>
      </div>
    </footer>
  );
}
