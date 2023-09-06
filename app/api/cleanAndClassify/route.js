import { NextResponse } from "next/server";

export async function POST(req) {
  const data = await req.json();

  const apiUrl = `${process.env.BACKEND_API}/runClassify`;
 
  try {
    const response = await fetch(apiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(`Failed to fetch data from the API. Status: ${response.status}, Message: ${errorData.message}`);
    }

    const responseData = await response.json();
    return NextResponse.json(responseData);
  } catch (error) {
    console.error("Error fetching sheet data:", error);
    return NextResponse.error({ status: 500, message: "Error fetching sheet data", error: error.message });
  }
}