import "./globals.css";
import { Inter } from "next/font/google";
import Header from "../components/Header";
import Footer from "../components/Footer";
import { UserProvider } from "@auth0/nextjs-auth0/client";
import PHProvider from "./components/PostHogProvider";

const inter = Inter({ subsets: ["latin"] });

export const metadata = {
  metadataBase: new URL('https://www.expensesorted.com/'),
  title: "Expense Sorted - Categorise your expenses",
  description: "Automatically categorise your monthly expenses using AI. Hook this App up to your Google Sheets™ and get your monthly budgeting done in no time.",
  alternates: {
    canonical: '/',
  },
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <UserProvider>
        <PHProvider>
          <body className={`${inter.className} flex flex-col min-h-screen`}>
            <Header />
            <div className="flex-grow">{children}</div>
            <Footer />
          </body>
        </PHProvider>
      </UserProvider>
    </html>
  );
}
