import { NextResponse } from 'next/server';
import { db } from '../../../db';
import { account } from '../../../db/schema';
import { eq } from 'drizzle-orm';

/**
 * GET handler for retrieving account information
 */
export async function GET(request: Request) {
  try {
    // Check if db is initialized
    if (!db) {
      return NextResponse.json({ error: 'Database connection not initialized' }, { status: 500 });
    }
    
    // Get userId from query parameters
    const { searchParams } = new URL(request.url);
    const userId = searchParams.get('userId');
    
    if (userId) {
      // Query a specific account
      const accounts = await db.select().from(account).where(eq(account.userId, userId));
      
      if (accounts.length === 0) {
        return NextResponse.json({ error: 'Account not found' }, { status: 404 });
      }
      
      return NextResponse.json(accounts[0]);
    } else {
      // Query all accounts
      const accounts = await db.select().from(account);
      return NextResponse.json(accounts);
    }
  } catch (error: any) {
    console.error('Error fetching account:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
}

/**
 * POST handler for creating or updating account information
 */
export async function POST(request: Request) {
  try {
    // Check if db is initialized
    if (!db) {
      return NextResponse.json({ error: 'Database connection not initialized' }, { status: 500 });
    }
    
    const body = await request.json();
    
    if (!body.userId) {
      return NextResponse.json({ error: 'userId is required' }, { status: 400 });
    }
    
    // Check if account exists
    const existingAccounts = await db.select({ userId: account.userId })
      .from(account)
      .where(eq(account.userId, body.userId));
    
    if (existingAccounts.length > 0) {
      // Update existing account
      await db.update(account)
        .set({
          categorisationRange: body.categorisationRange,
          categorisationTab: body.categorisationTab,
          columnOrderCategorisation: body.columnOrderCategorisation,
          apiKey: body.apiKey || body.api_key, // Support both formats
        })
        .where(eq(account.userId, body.userId));
    } else {
      // Create new account
      await db.insert(account).values({
        userId: body.userId,
        categorisationRange: body.categorisationRange,
        categorisationTab: body.categorisationTab,
        columnOrderCategorisation: body.columnOrderCategorisation,
        apiKey: body.apiKey || body.api_key, // Support both formats
      });
    }
    
    return NextResponse.json({ success: true, data: body });
  } catch (error: any) {
    console.error('Error creating/updating account:', error);
    return NextResponse.json({ error: error.message }, { status: 500 });
  }
} 