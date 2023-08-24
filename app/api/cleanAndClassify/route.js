import { NextResponse } from "next/server";

export async function POST(req) {
  // const data = await req.json();
  const formData  = await req.formData();
  console.log("🚀 ~ file: route.js:6 ~ POST ~ formData:", formData)

  const apiUrl = "https://555d-120-88-75-106.ngrok-free.app/runClassify";
  try {
    const response = await fetch(apiUrl, {
      method: "POST",
      body: formData,
    });
    // console.log("🚀 ~ file: route.js:13 ~ POST ~ response:", response)

    if (!response.ok) {
      throw new Error("Failed to fetch data from the API");
    }

    const responseData = await response.json();
    // console.log("🚀 ~ file: route.js:20 ~ POST ~ responseData:", responseData)
    return NextResponse.json(responseData);
  } catch (error) {
    console.error("Error fetching sheet data:", error);
    return NextResponse.error("Error fetching sheet data");
  }
}