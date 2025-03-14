import { NextResponse } from "next/server";

export async function POST(req) {
  try {
    const data = await req.json();
    const email = data.email;
    const tags = data.tags || []; // Get tags if provided
    const lists = data.lists || ["441356"]; // Get lists if provided, default to landing page list

    if (!email) {
      return NextResponse.json({ error: "Email is required" }, { status: 400 });
    }

    // Prepare the request body
    const requestBody = {
      email,
      lists: lists, // Use the provided lists or default
    };

    // Add tags if provided
    if (tags.length > 0) {
      requestBody.tags = tags;
    }

    const response = await fetch("https://api.sendfox.com/contacts", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${process.env.SENDFOX_API_KEY}`,
      },
      body: JSON.stringify(requestBody),
    });

    // Get the response data
    const responseData = await response.json().catch(() => null);

    if (response.ok) {
      return NextResponse.json({
        success: true,
        message: "Successfully subscribed!",
      });
    } else {
      // Check if it's a duplicate email error (which is actually okay for us)
      if (responseData?.message?.includes("already exists")) {
        // For existing contacts, we might want to update their tags
        // This would require an additional API call to update the contact
        // But for simplicity, we'll just return success
        return NextResponse.json({
          success: true,
          message: "Email already subscribed",
        });
      }

      console.error("SendFox API error:", responseData);
      return NextResponse.json(
        {
          error: "Failed to subscribe. Please try again.",
        },
        { status: response.status }
      );
    }
  } catch (error) {
    console.error("Error in email subscription:", error);
    return NextResponse.json({ error: "An unexpected error occurred" }, { status: 500 });
  }
}
