import { NextResponse } from "next/server";

export async function POST(req) {
  try {
    const data = await req.json();

    const apiUrl = `${process.env.BACKEND_API}/runTraining`;
  
    console.log("🚀 ~ POST ~ apiUrl:", apiUrl);
    const response = await fetch(apiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      console.error(`Error: ${response.status} ${response.statusText}`);
      return NextResponse.json({ error: `${response.status} ${response.statusText}` }, { status: response.status });
    }

    const responseData = await response.json();
    return NextResponse.json(responseData);
  } catch (error) {
    console.error("Error fetching sheet data:", error);
    return NextResponse.json({ error: `Error fetching sheet data: ${error.message}` }, { status: 500 });
  }
}