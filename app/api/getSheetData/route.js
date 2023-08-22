import { NextResponse } from "next/server";
import { google } from "googleapis"; // Import Google Sheets API library
import fs from "fs";
import os from "os";
import path from "path";

export async function POST(req) {
  try {
    const formData = await req.formData();
    const credentialsFile = formData.get("credentialsFile");
    const spreadsheetId = formData.get("spreadsheetId");
    let range = formData.get("range");

    if (!credentialsFile || !spreadsheetId || !range) {
      return NextResponse.json({ error: "Missing parameters" }, { status: 400 });
    }

    const credentialsContent = await credentialsFile.text();
    const tempDir = fs.mkdtempSync(path.join(os.tmpdir(), "txClassify-"));
    const credentialsPath = path.join(tempDir, "temporaryCredentials.json");

    // Write the credentials content to the temporary path
    fs.writeFileSync(credentialsPath, credentialsContent);

    const credentials = JSON.parse(fs.readFileSync(credentialsPath, "utf8"));
    const auth = new google.auth.GoogleAuth({
      credentials,
      scopes: ["https://www.googleapis.com/auth/spreadsheets"],
    });

    const sheets = google.sheets({ version: "v4", auth });

    range = `Details!${range}`;

    const response = await sheets.spreadsheets.values.get({
      spreadsheetId,
      range,
    });

    const sheetData = response.data.values;
    const cleanedSheetData = sheetData.flat().filter((item) => item !== "");

    // Clean up by removing the temporary directory
    fs.rmSync(tempDir, { recursive: true, force: true });

    return NextResponse.json(cleanedSheetData);
  } catch (error) {
    console.error("Error fetching sheet data:", error);
    return NextResponse.error("Error fetching sheet data");
  }
}
