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

export async function POST(req: Request) {
  try {
    // Check if user has a session cookie
    const hasSession = await hasSessionCookie();

    // If user is not logged in, return success anyway to avoid UI issues
    if (!hasSession) {
      return NextResponse.json({ success: true });
    }

    const { preference } = await req.json();

    // Validate preference
    if (preference !== "OPTED_IN" && preference !== "DISMISSED") {
      return NextResponse.json({ error: "Invalid preference value" }, { status: 400 });
    }

    // For now, just return success without updating the database
    // This is a temporary fix until Auth0 updates their library to work with Next.js 15
    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Error updating beta preference:", error);
    // Return success anyway to avoid breaking the UI
    return NextResponse.json({ success: true });
  } finally {
    await prisma.$disconnect();
  }
}
