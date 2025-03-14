import { NextResponse } from "next/server";
import { getSession } from "@auth0/nextjs-auth0";
import { PrismaClient } from "@prisma/client";

const prisma = new PrismaClient();

export async function POST(req: Request) {
  try {
    // Use getSession without parameters
    const session = await getSession();

    // If user is not logged in, return error
    if (!session || !session.user) {
      return NextResponse.json({ error: "User not authenticated" }, { status: 401 });
    }

    const userId = session.user.sub;
    const { preference } = await req.json();

    // Validate preference
    if (preference !== "OPTED_IN" && preference !== "DISMISSED") {
      return NextResponse.json({ error: "Invalid preference value" }, { status: 400 });
    }

    // Update the user's account in the database
    await prisma.account.update({
      where: { userId },
      data: { appBetaOptIn: preference } as any,
    });

    return NextResponse.json({ success: true });
  } catch (error) {
    console.error("Error updating beta preference:", error);
    return NextResponse.json({ error: "Failed to update beta preference" }, { status: 500 });
  } finally {
    await prisma.$disconnect();
  }
}
