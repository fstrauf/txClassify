import { NextResponse } from "next/server";
import { PrismaClient } from "@prisma/client";
import { cookies } from "next/headers";

const prisma = new PrismaClient();

// Simple function to check if user has a session cookie
async function hasSessionCookie() {
  try {
    // In Next.js 15, cookies() returns a Promise that must be awaited
    const cookieStore = await cookies();
    const sessionCookie = cookieStore.get("appSession");

    return sessionCookie?.value ? true : false;
  } catch (error) {
    console.error("Error checking session cookie:", error);
    return false;
  }
}

export async function GET(req: Request) {
  try {
    // Check if user has a session cookie
    const hasSession = await hasSessionCookie();

    // If user is not logged in, return null preference
    if (!hasSession) {
      return NextResponse.json({ preference: null });
    }

    // For now, just return DISMISSED to ensure the banner doesn't show
    // This is a temporary fix until Auth0 updates their library to work with Next.js 15
    return NextResponse.json({ preference: "DISMISSED" });
  } catch (error) {
    console.error("Error getting beta preference:", error);
    // Return DISMISSED preference on error to avoid showing the banner
    return NextResponse.json({ preference: "DISMISSED" });
  } finally {
    await prisma.$disconnect();
  }
}
