import { NextResponse } from "next/server";
import Replicate from "replicate";

export async function POST(req) {
  const data = await req.json();
  try {
    const replicate = new Replicate({
      // get your token from https://replicate.com/account
      auth: process.env.REPLICATE_API_TOKEN,
    });

    let prediction = await replicate.predictions.create({
      version:
        "b6b7585c9640cd7a9572c6e129c9549d79c9c31f0d3fdce7baac7c67ca38f305",
      input: {
        text_batch: JSON.stringify(data?.training_data),
      },
      webhook: `https://pythonhandler-yxxxtrqkpa-ts.a.run.app/${data?.apiMode}?customerName=${data?.customerName}&sheetApi=${data?.sheetApi}`,
      webhook_events_filter: ["completed"],
    });
    return NextResponse.json(prediction);
  } catch (error) {
    console.error("Error fetching sheet data:", error);
    return NextResponse.error("Error fetching sheet data");
  }
}
