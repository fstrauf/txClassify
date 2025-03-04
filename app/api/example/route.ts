import { NextResponse } from 'next/server';
import { db } from '@/db';
import { account, webhookResults } from '@/db/schema';
import { eq } from 'drizzle-orm';

export async function GET() {
  try {
    // Example: Get all accounts
    if (!db) {
      return NextResponse.json(
        { success: false, error: 'Database connection not initialized' },
        { status: 500 }
      );
    }
    
    const accounts = await db.select().from(account);
    
    return NextResponse.json({ 
      success: true, 
      data: accounts 
    });
  } catch (error) {
    console.error('Database query error:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to fetch data from database' },
      { status: 500 }
    );
  }
}

export async function POST(request: Request) {
  try {
    if (!db) {
      return NextResponse.json(
        { success: false, error: 'Database connection not initialized' },
        { status: 500 }
      );
    }
    
    const body = await request.json();
    
    // Example: Insert a new account
    const result = await db.insert(account).values({
      userId: body.userId,
      categorisationRange: body.categorisationRange || 'A:Z',
      categorisationTab: body.categorisationTab || 'Sheet1',
      columnOrderCategorisation: body.columnOrderCategorisation || { categoryColumn: 'E', descriptionColumn: 'C' },
      apiKey: body.apiKey || '',
    }).returning();
    
    return NextResponse.json({ 
      success: true, 
      data: result[0] 
    });
  } catch (error) {
    console.error('Database insert error:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to insert data into database' },
      { status: 500 }
    );
  }
}

export async function PUT(request: Request) {
  try {
    if (!db) {
      return NextResponse.json(
        { success: false, error: 'Database connection not initialized' },
        { status: 500 }
      );
    }
    
    const body = await request.json();
    
    if (!body.userId) {
      return NextResponse.json(
        { success: false, error: 'userId is required' },
        { status: 400 }
      );
    }
    
    // Example: Update an account
    const result = await db.update(account)
      .set({
        categorisationRange: body.categorisationRange,
        categorisationTab: body.categorisationTab,
        columnOrderCategorisation: body.columnOrderCategorisation,
        apiKey: body.apiKey,
      })
      .where(eq(account.userId, body.userId))
      .returning();
    
    if (result.length === 0) {
      return NextResponse.json(
        { success: false, error: 'Account not found' },
        { status: 404 }
      );
    }
    
    return NextResponse.json({ 
      success: true, 
      data: result[0] 
    });
  } catch (error) {
    console.error('Database update error:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to update data in database' },
      { status: 500 }
    );
  }
}

export async function DELETE(request: Request) {
  try {
    if (!db) {
      return NextResponse.json(
        { success: false, error: 'Database connection not initialized' },
        { status: 500 }
      );
    }
    
    const { searchParams } = new URL(request.url);
    const userId = searchParams.get('userId');
    
    if (!userId) {
      return NextResponse.json(
        { success: false, error: 'userId is required' },
        { status: 400 }
      );
    }
    
    // Example: Delete an account
    const result = await db.delete(account)
      .where(eq(account.userId, userId))
      .returning();
    
    if (result.length === 0) {
      return NextResponse.json(
        { success: false, error: 'Account not found' },
        { status: 404 }
      );
    }
    
    return NextResponse.json({ 
      success: true, 
      data: result[0] 
    });
  } catch (error) {
    console.error('Database delete error:', error);
    return NextResponse.json(
      { success: false, error: 'Failed to delete data from database' },
      { status: 500 }
    );
  }
} 