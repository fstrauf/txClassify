import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/prisma';
import { Prisma } from '@prisma/client';

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url);
    const userId = searchParams.get('userId');

    if (!userId) {
      return NextResponse.json({ error: 'User ID is required' }, { status: 400 });
    }

    const account = await prisma.account.findUnique({
      where: { userId },
      select: { api_key: true }
    });

    return NextResponse.json(account || { api_key: null });
  } catch (error) {
    console.error('Error fetching account:', error);
    
    // Handle specific Prisma errors
    if (error instanceof Prisma.PrismaClientKnownRequestError) {
      if (error.code === 'P2002') {
        return NextResponse.json({ error: 'A unique constraint would be violated.' }, { status: 409 });
      }
      if (error.code === 'P2025') {
        return NextResponse.json({ error: 'Record not found.' }, { status: 404 });
      }
    }
    
    // Handle connection errors
    if (error instanceof Prisma.PrismaClientInitializationError) {
      return NextResponse.json({ error: 'Database connection failed.' }, { status: 503 });
    }
    
    return NextResponse.json({ error: 'Failed to fetch account' }, { status: 500 });
  }
}

export async function POST(request: NextRequest) {
  try {
    const body = await request.json();
    const { userId, api_key } = body;

    if (!userId) {
      return NextResponse.json({ error: 'User ID is required' }, { status: 400 });
    }

    const account = await prisma.account.upsert({
      where: { userId },
      update: { api_key },
      create: { 
        userId,
        api_key,
        categorisationRange: null,
        categorisationTab: null,
        columnOrderCategorisation: Prisma.JsonNull
      }
    });

    return NextResponse.json(account);
  } catch (error) {
    console.error('Error updating account:', error);
    
    // Handle specific Prisma errors
    if (error instanceof Prisma.PrismaClientKnownRequestError) {
      if (error.code === 'P2002') {
        return NextResponse.json({ error: 'A unique constraint would be violated.' }, { status: 409 });
      }
    }
    
    // Handle connection errors
    if (error instanceof Prisma.PrismaClientInitializationError) {
      return NextResponse.json({ error: 'Database connection failed.' }, { status: 503 });
    }
    
    return NextResponse.json({ error: 'Failed to update account' }, { status: 500 });
  }
} 