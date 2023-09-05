import { NextResponse } from 'next/server';
import jwt from 'jsonwebtoken';

export async function POST(req) {
  const data = await req.json();
  console.log("🚀 ~ file: route.js:5 ~ POST ~ data:", data.value)

  try {
    const client_email = process.env.GOOGLE_SERVICE_ACCOUNT_EMAIL
    const private_key = process.env.GOOGLE_SERVICE_PRIVATE_KEY

    // Create a JWT
    const token = jwt.sign({
      iss: client_email,
      scope: 'https://www.googleapis.com/auth/spreadsheets',
      aud: 'https://oauth2.googleapis.com/token',
      exp: Math.floor(Date.now() / 1000) + (60 * 60),  // 1 hour
      iat: Math.floor(Date.now() / 1000)
    }, private_key, { algorithm: 'RS256' });

    // Exchange the JWT for an access token
    const res = await fetch('https://oauth2.googleapis.com/token', {
      method: 'POST',
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      body: new URLSearchParams({
        grant_type: 'urn:ietf:params:oauth:grant-type:jwt-bearer',
        assertion: token
      })
    });

    const { access_token } = await res.json();

    // Use the access token to call the Google Sheets API
    const response = await fetch(
      `https://sheets.googleapis.com/v4/spreadsheets/${data.expenseSheetId}`,
      {
        headers: {
          Authorization: `Bearer ${access_token}`,
        },
      }
    );

    if (response.ok) {
      const resData = await response.json();
      const spreadsheetName = resData.properties.title;
      const sheetName = resData.sheets[0].properties.title;
      return NextResponse.json({ spreadsheetName, sheetName });
    } else {
      console.error("Error fetching spreadsheet data:", response.statusText);
      return NextResponse.error("Error fetching spreadsheet data");
    }
  } catch (error) {
    console.error("Error Fetching Sheet Data:", error);
    return NextResponse.error("Error Fetching Sheet Data");
  }
}