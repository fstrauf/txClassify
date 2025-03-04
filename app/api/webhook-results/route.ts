import { NextResponse } from 'next/server';
import { db } from '../../../db';
import { webhookResults } from '../../../db/schema';
import { eq } from 'drizzle-orm';

/**
 * GET handler for retrieving webhook results
 */
export async function GET(request: Request) {
  try {
    // Check if db is initialized
    if (!db) {
      return NextResponse.json({ error: 'Database connection not initialized' }, { status: 500 });
    }
    
    // Get predictionId from query parameters
    const { searchParams } = new URL(request.url);
    const predictionId = searchParams.get('predictionId');
    
    if (!predictionId) {
      return NextResponse.json({ error: 'predictionId is required' }, { status: 400 });
    }
    
    // Query the webhook results using Drizzle
    const results = await db.select()
      .from(webhookResults)
      .where(eq(webhookResults.predictionId, predictionId));
    
    if (results.length === 0) {
      return NextResponse.json({ error: 'Webhook result not found' }, { status: 404 });
    }
    
    return NextResponse.json(results[0]);
  } catch (error: any) {
    console.error('Error fetching webhook result:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}

/**
 * POST handler for creating webhook results
 */
export async function POST(request: Request) {
  try {
    // Check if db is initialized
    if (!db) {
      return NextResponse.json({ error: 'Database connection not initialized' }, { status: 500 });
    }
    
    const body = await request.json();
    
    if (!body.prediction_id) {
      return NextResponse.json({ error: 'prediction_id is required' }, { status: 400 });
    }
    
    if (!body.results) {
      return NextResponse.json({ error: 'results are required' }, { status: 400 });
    }
    
    // Create new webhook result using Drizzle
    const result = await db.insert(webhookResults).values({
      predictionId: body.prediction_id,
      results: body.results,
    }).returning();
    
    return NextResponse.json({ success: true, data: result[0] });
  } catch (error: any) {
    console.error('Error creating webhook result:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}

/**
 * DELETE handler for removing webhook results
 */
export async function DELETE(request: Request) {
  try {
    // Check if db is initialized
    if (!db) {
      return NextResponse.json({ error: 'Database connection not initialized' }, { status: 500 });
    }
    
    // Get predictionId from query parameters
    const { searchParams } = new URL(request.url);
    const predictionId = searchParams.get('predictionId');
    
    if (!predictionId) {
      return NextResponse.json({ error: 'predictionId is required' }, { status: 400 });
    }
    
    // Delete the webhook result using Drizzle
    await db.delete(webhookResults)
      .where(eq(webhookResults.predictionId, predictionId));
    
    return NextResponse.json({ success: true });
  } catch (error: any) {
    console.error('Error deleting webhook result:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
} 