import { NextResponse } from "next/server";

export async function POST(req) {
  const data = await req.json();

  const email = data.email

  try {
    // Create a new contact in SendFox using the Fetch API
    const response = await fetch("https://api.sendfox.com/contacts", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${process.env.SENDFOX_API_KEY}`,
      },
      body: JSON.stringify({ email, lists: ["441356"] }), // You can add more properties if needed
    });

    if (response.ok) {
      return NextResponse.json(response.json());
    } else {
      console.error("Error creating contact:", response.statusText);
    }
  } catch (error) {
    console.error("Error submitting email:", error);
    return NextResponse.error("Error submitting email");
  }

}

