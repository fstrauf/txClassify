# TxClassify

## Introduction

This is a hybrid Next.js + Python app that uses Next.js as the frontend and Flask as the API backend. One great use case of this is to write Next.js apps that use Python AI libraries on the backend.

## Database Layer

The application uses Drizzle ORM with Supabase PostgreSQL for database operations. The database layer is implemented in the `db/` directory.

### Key Features

- **Schema Definition**: Database schema is defined using Drizzle ORM's schema definition language
- **Type-Safe Queries**: Full TypeScript support for database operations with compile-time type checking
- **Migrations**: Database migrations are managed using Drizzle Kit
- **Direct Integration**: Direct connection to Supabase PostgreSQL database

### Architecture

The database layer is implemented using a clean architecture:

1. **Server-side (API Routes)**: Uses Drizzle ORM with postgres-js adapter for type-safe database operations
2. **Client-side Components**: Use API routes to access the database, maintaining type safety
3. **Schema Definitions**: Shared between server and client for consistent types

This approach allows us to use Drizzle ORM throughout the application while avoiding browser compatibility issues with Node.js-specific modules.

### Setup

To set up the database connection:

1. Go to your Supabase project dashboard
2. Navigate to Project Settings > Database
3. Find the "Connection string" section and select "URI" format
4. Copy the connection string that looks like:
   ```
   postgres://postgres.goakphmfcayylqvnigvv:your-password@aws-0-us-west-1.pooler.supabase.com:5432/postgres
   ```
5. Replace `your-password` with your actual database password
6. Add this connection string to your `.env.local` file as:
   ```
   DATABASE_URL=your_connection_string_here
   ```

### Example Usage

```typescript
import { db } from '../db';
import { account } from '../db/schema';
import { eq } from 'drizzle-orm';

// Query the database
const users = await db.select().from(account);

// Insert data
await db.insert(account).values({
  userId: 'user123',
  apiKey: 'api-key-123',
});
```

For more information, see the [Database Layer README](./db/README.md).

## How It Works

```npm run dev``` spins up flask with python and nextjs.

```python3 api/index.py``` let's you test the API separately as a python script

### Database Commands

- Setup database connection: `pnpm db:setup`
- Generate migrations: `pnpm db:generate`
- Apply migrations: `pnpm db:migrate`
- Introspect database: `pnpm db:introspect`
- Open Drizzle Studio: `pnpm db:studio`

### API Examples

```bash
curl -m 70 -X POST "http://localhost:8080/txclassify?mode=train" \
-H "Authorization: bearer $(gcloud auth print-identity-token)" \
-F "file=@/Users/fstrauf/Documents/01_code/txClassify/all_expenses_classified.csv"

curl -m 70 -X POST "https://us-central1-txclassify.cloudfunctions.net/txclassify?mode=train" \
-H "Authorization: bearer $(gcloud auth print-identity-token)" \
-F "file=@/Users/fstrauf/Documents/01_code/txClassify/all_expenses_classified.csv"
```

