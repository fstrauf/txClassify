# Database Layer with Drizzle ORM

This directory contains the database layer implementation using Drizzle ORM with Supabase PostgreSQL.

## Directory Structure

- `db/schema/`: Contains the database schema definitions
- `db/migrations/`: Contains the generated migration files
- `db/index.ts`: Main entry point for database operations
- `db/migrate.ts`: Script to run migrations

## Schema

The database schema is defined using Drizzle ORM's schema definition language. The schema files are located in the `db/schema/` directory.

- `account.ts`: Schema for the account table
- `webhook_results.ts`: Schema for the webhook_results table
- `index.ts`: Exports all schema definitions

## Database Operations

The database operations are implemented using Drizzle ORM's query builder. Here's an example of how to use it:

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

// Update an account
await db.update(account)
  .set({ apiKey: 'new-api-key-123' })
  .where(eq(account.userId, 'user123'));

// Delete an account
await db.delete(account)
  .where(eq(account.userId, 'user123'));
```

## Migrations

Database migrations are managed using Drizzle Kit. The migration files are generated in the `db/migrations/` directory.

### Generate Migrations

To generate migrations based on schema changes:

```bash
pnpm db:generate
```

### Apply Migrations

To apply migrations to the database:

```bash
pnpm db:migrate
```

### Introspect Database

To introspect an existing database and generate schema:

```bash
pnpm db:introspect
```

### Drizzle Studio

To view and manage your database using Drizzle Studio:

```bash
pnpm db:studio
```

## Environment Variables

The database connection requires the following environment variables:

- `SUPABASE_DB_HOST`: Supabase database host
- `SUPABASE_DB_PORT`: Supabase database port
- `SUPABASE_DB_USER`: Supabase database user
- `SUPABASE_DB_PASSWORD`: Supabase database password
- `SUPABASE_DB_NAME`: Supabase database name

Alternatively, you can use a connection string:

- `DATABASE_URL`: Full database connection string

## Usage in API Routes

Example of using the database layer in an API route:

```typescript
import { select, insert, update, deleteFrom } from '../../db';

// Query data
const response = await select('account', ['*'], { userId: 'user123' });

// Insert data
const insertResponse = await insert('account', { userId: 'user123', api_key: 'key123' });

// Update data
const updateResponse = await update('account', { api_key: 'newkey123' }, { userId: 'user123' });

// Delete data
const deleteResponse = await deleteFrom('account', { userId: 'user123' });
``` 