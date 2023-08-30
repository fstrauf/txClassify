import { NextResponse } from "next/server";

export async function POST(req) {
  try {
    // const data = await req.json();
    const formData  = await req.formData();
    console.log("🚀 ~ file: route.js:6 ~ POST ~ formData:", formData)

    const apiUrl = `${process.env.BACKEND_API}/runTraining`;
  
    const response = await fetch(apiUrl, {
      method: "POST",
      body: formData,
    });
    // console.log("🚀 ~ file: route.js:13 ~ POST ~ response:", response)

    if (!response.ok) {
      console.error(`Error: ${response.status} ${response.statusText}`);
      return NextResponse.error(`Error: ${response.status} ${response.statusText}`);
    }

    const responseData = await response.json();
    // console.log("🚀 ~ file: route.js:20 ~ POST ~ responseData:", responseData)
    return NextResponse.json(responseData);
  } catch (error) {
    console.error("Error fetching sheet data:", error);
    return NextResponse.error(`Error fetching sheet data: ${error.message}`);
  }
}