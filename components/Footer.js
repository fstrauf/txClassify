// components/Footer.js

import Link from "next/link";

export default function Footer() {
  return (
    <footer className="w-full bg-surface border-t border-gray-200">
      <div className="container mx-auto px-4 py-8">
        <div className="flex flex-col md:flex-row justify-between items-center gap-4">
          <div className="flex flex-wrap gap-4 justify-center md:justify-start">
            <Link 
              href="/privacy-policy" 
              className="text-gray-600 hover:text-primary transition-colors text-sm"
            >
              Privacy Policy
            </Link>
            <Link 
              href="/support" 
              className="text-gray-600 hover:text-primary transition-colors text-sm"
            >
              Support
            </Link>
            <Link 
              href="/terms-of-service" 
              className="text-gray-600 hover:text-primary transition-colors text-sm"
            >
              Terms of Service
            </Link>
            <Link 
              href="/about" 
              className="text-gray-600 hover:text-primary transition-colors text-sm"
            >
              About
            </Link>
          </div>
          <div className="text-gray-500 text-sm">
            © {new Date().getFullYear()} Expense Sorted. All rights reserved.
          </div>
        </div>
      </div>
    </footer>
  );
}
