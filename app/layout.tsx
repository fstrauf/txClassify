import "./globals.css";
import { Inter } from "next/font/google";
import Header from "../components/Header";
import Footer from "../components/Footer";
import PostHogProviderWrapper from "./components/PostHogProvider";
import { auth0 } from "@/src/lib/auth0";
import { Auth0ClientProvider } from "@/components/Auth0ClientProvider";

const inter = Inter({ subsets: ["latin"] });

export const metadata = {
  metadataBase: new URL("https://www.expensesorted.com/"),
  title: "Expense Sorted - Categorise your expenses",
  description:
    "Automatically categorise your monthly expenses using AI. Hook this App up to your Google Sheets™ and get your monthly budgeting done in no time.",
  alternates: {
    canonical: "/",
  },
};

export default async function RootLayout({ children }: { children: React.ReactNode }) {
  const session = await auth0.getSession();
  const user = session?.user;

  return (
    <html lang="en">
      <body className={`${inter.className} flex flex-col min-h-screen`}>
        <PostHogProviderWrapper>
          <Auth0ClientProvider user={user}>
            <Header />
            <div className="flex-grow">{children}</div>
            <Footer />
          </Auth0ClientProvider>
        </PostHogProviderWrapper>
      </body>
    </html>
  );
}
