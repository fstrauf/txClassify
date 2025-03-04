# Drizzle ORM Integration

This document summarizes the integration of Drizzle ORM into the TxClassify project.

## What We've Accomplished

1. **Schema Definition**
   - Created schema definitions for existing database tables (`account` and `webhook_results`)
   - Defined relationships and constraints based on the existing database structure

2. **Database Client**
   - Set up a direct connection to the Supabase PostgreSQL database using Drizzle ORM
   - Configured the database client to work in server-side environments only
   - Implemented a clean architecture with API routes for client-side access

3. **Migration Framework**
   - Set up Drizzle Kit for managing database migrations
   - Created scripts for generating and applying migrations
   - Added a setup script to help configure database connection details

4. **API Integration**
   - Updated API routes to use Drizzle ORM for database operations
   - Implemented type-safe queries with proper error handling
   - Created a clean separation between client and server code

5. **Frontend Integration**
   - Updated components to work with the API routes
   - Created an example component to demonstrate database operations
   - Maintained type safety throughout the application

## Benefits

- **Type Safety**: Full TypeScript support for database operations
- **Query Building**: Intuitive and powerful query building API
- **Migration Management**: Easy-to-use migration system for schema changes
- **Performance**: Efficient database operations with prepared statements
- **Developer Experience**: Better developer experience with autocomplete and type checking
- **Clean Architecture**: Clear separation between client and server code

## Architecture

The database layer is implemented using a clean architecture:

1. **Server-side (API Routes)**: Uses Drizzle ORM with postgres-js adapter for type-safe database operations
2. **Client-side Components**: Use API routes to access the database, maintaining type safety
3. **Schema Definitions**: Shared between server and client for consistent types

This approach allows us to use Drizzle ORM throughout the application while avoiding browser compatibility issues with Node.js-specific modules.

## Next Steps

1. **Complete Migration**: Migrate all remaining database operations to use Drizzle ORM
2. **Add Relations**: Define relationships between tables for more complex queries
3. **Optimize Queries**: Optimize database queries for better performance
4. **Add Indexes**: Add indexes to improve query performance
5. **Add Validation**: Add validation for database operations

## Usage Examples

### Server-side (API Routes)

```typescript
import { db } from '../db';
import { account } from '../db/schema';
import { eq } from 'drizzle-orm';

// Select all accounts
const allAccounts = await db.select().from(account);

// Select a specific account
const specificAccount = await db.select()
  .from(account)
  .where(eq(account.userId, 'user123'));

// Insert a new account
await db.insert(account).values({
  userId: 'user123',
  categorisationRange: 'A:Z',
  categorisationTab: 'Sheet1',
  columnOrderCategorisation: { categoryColumn: 'E', descriptionColumn: 'C' },
  apiKey: 'api-key-123',
});
```

### Client-side (Components)

```typescript
// Fetch data from API route
const fetchAccounts = async () => {
  const response = await fetch('/api/account');
  const accounts = await response.json();
  return accounts;
};

// Create data using API route
const createAccount = async (accountData) => {
  const response = await fetch('/api/account', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(accountData),
  });
  return await response.json();
};
``` 