import { NextResponse } from "next/server";

export async function POST(req) {
  const data = await req.json();

  const email = data.email;

  try {
    const response = await fetch("https://api.sendfox.com/contacts", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${process.env.SENDFOX_API_KEY}`,
      },
      body: JSON.stringify({ email, lists: ["441356"] }),
    });

    if (response.ok) {
      const jsonResponse = await response.json();
      return NextResponse.json(jsonResponse);
    } else {
      console.error("Error creating contact:", response.statusText);
      return NextResponse.error("Error creating contact");
    }
  } catch (error) {
    console.error("Error submitting email:", error);
    return NextResponse.error("Error submitting email");
  }
}
