import { NextResponse } from "next/server";

export async function POST(req) {
  // const data = await req.json();
  const formData  = await req.formData();
  console.log("🚀 ~ file: route.js:6 ~ POST ~ formData:", formData)

  const apiUrl = "https://3efe-65-181-3-157.ngrok-free.app/runTraining";
  // const apiUrl = "https://pythonhandler-yxxxtrqkpa-ts.a.run.app/runTraining"
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