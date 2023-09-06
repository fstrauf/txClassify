import { NextResponse } from "next/server";

export async function POST(req) {
  try {
    const data = await req.json();

    const apiUrl = `${process.env.BACKEND_API}/runTraining`;
  
    const response = await fetch(apiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      console.error(`Error: ${response.status} ${response.statusText}`);
      return NextResponse.error(`Error: ${response.status} ${response.statusText}`);
    }

    const responseData = await response.json();
    return NextResponse.json(responseData);
  } catch (error) {
    console.error("Error fetching sheet data:", error);
    return NextResponse.error(`Error fetching sheet data: ${error.message}`);
  }
}