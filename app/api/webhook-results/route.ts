import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/prisma';
import { Prisma } from '@prisma/client';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const predictionId = searchParams.get('prediction_id');

    if (!predictionId) {
      return NextResponse.json({ error: 'Prediction ID is required' }, { status: 400 });
    }

    const webhookResult = await prisma.webhookResult.findUnique({
      where: { prediction_id: predictionId }
    });

    if (!webhookResult) {
      return NextResponse.json({ error: 'Webhook result not found' }, { status: 404 });
    }

    return NextResponse.json(webhookResult);
  } catch (error) {
    console.error('Error fetching webhook result:', error);
    
    // Handle specific Prisma errors
    if (error instanceof Prisma.PrismaClientKnownRequestError) {
      if (error.code === 'P2025') {
        return NextResponse.json({ error: 'Record not found.' }, { status: 404 });
      }
    }
    
    // Handle connection errors
    if (error instanceof Prisma.PrismaClientInitializationError) {
      return NextResponse.json({ error: 'Database connection failed.' }, { status: 503 });
    }
    
    return NextResponse.json({ error: 'Failed to fetch webhook result' }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { prediction_id, results } = body;

    if (!prediction_id || !results) {
      return NextResponse.json({ error: 'Prediction ID and results are required' }, { status: 400 });
    }

    // Check if a webhook result with this prediction_id already exists
    const existingResult = await prisma.webhookResult.findUnique({
      where: { prediction_id }
    });

    if (existingResult) {
      return NextResponse.json({ error: 'Webhook result already exists for this prediction ID' }, { status: 409 });
    }

    const webhookResult = await prisma.webhookResult.create({
      data: {
        prediction_id,
        results
      }
    });

    return NextResponse.json(webhookResult);
  } catch (error) {
    console.error('Error creating webhook result:', error);
    
    // Handle specific Prisma errors
    if (error instanceof Prisma.PrismaClientKnownRequestError) {
      if (error.code === 'P2002') {
        return NextResponse.json({ error: 'A webhook result with this prediction ID already exists.' }, { status: 409 });
      }
    }
    
    // Handle connection errors
    if (error instanceof Prisma.PrismaClientInitializationError) {
      return NextResponse.json({ error: 'Database connection failed.' }, { status: 503 });
    }
    
    return NextResponse.json({ error: 'Failed to create webhook result' }, { status: 500 });
  }
} 