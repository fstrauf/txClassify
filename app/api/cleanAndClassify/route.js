import { NextResponse } from "next/server";

export async function POST(req) {
  const data = await req.json();
  console.log("🚀 ~ Sending data to backend:", data);

  const apiUrl = `${process.env.BACKEND_API}/process_transactions`;
 
  try {
    const response = await fetch(apiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      let errorMessage;
      try {
        const errorData = await response.json();
        console.log("🚀 ~ Error response from backend:", errorData);
        errorMessage = errorData.message || errorData.error || 'Unknown error';
      } catch (e) {
        // If response is not JSON, get text instead
        errorMessage = await response.text();
        console.log("🚀 ~ Error text from backend:", errorMessage);
      }
      throw new Error(`Failed to fetch data from the API. Status: ${response.status}, Message: ${errorMessage}`);
    }

    const responseData = await response.json();
    return NextResponse.json(responseData);
  } catch (error) {
    console.error("Error fetching sheet data:", error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}