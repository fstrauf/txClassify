import { NextResponse } from "next/server";
import { getSession } from "@auth0/nextjs-auth0";
import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();

export async function GET(req: Request) {
  try {
    // Use getSession without parameters
    const session = await getSession();

    // If user is not logged in, return null preference
    if (!session || !session.user) {
      return NextResponse.json({ preference: null });
    }

    const userId = session.user.sub;

    // Get the user's account from the database
    const account = await prisma.account.findUnique({
      where: { userId },
      select: { appBetaOptIn: true } as any,
    });

    return NextResponse.json({ preference: (account as any)?.appBetaOptIn || null });
  } catch (error) {
    console.error("Error getting beta preference:", error);
    return NextResponse.json({ error: "Failed to get beta preference" }, { status: 500 });
  } finally {
    await prisma.$disconnect();
  }
}
