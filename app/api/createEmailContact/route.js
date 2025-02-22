import { NextResponse } from "next/server";

export async function POST(req) {
  try {
    const data = await req.json();
    const email = data.email;

    if (!email) {
      return NextResponse.json(
        { error: "Email is required" },
        { status: 400 }
      );
    }

    const response = await fetch("https://api.sendfox.com/contacts", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${process.env.SENDFOX_API_KEY}`,
      },
      body: JSON.stringify({ 
        email, 
        lists: ["441356"] 
      }),
    });

    // Get the response data
    const responseData = await response.json().catch(() => null);

    if (response.ok) {
      return NextResponse.json({ 
        success: true,
        message: "Successfully subscribed!" 
      });
    } else {
      // Check if it's a duplicate email error (which is actually okay for us)
      if (responseData?.message?.includes("already exists")) {
        return NextResponse.json({ 
          success: true,
          message: "Email already subscribed" 
        });
      }

      console.error("SendFox API error:", responseData);
      return NextResponse.json(
        { 
          error: "Failed to subscribe. Please try again." 
        },
        { status: response.status }
      );
    }
  } catch (error) {
    console.error("Error in email subscription:", error);
    return NextResponse.json(
      { error: "An unexpected error occurred" },
      { status: 500 }
    );
  }
}
