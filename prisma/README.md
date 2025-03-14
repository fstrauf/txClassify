# Prisma ORM Integration with Supabase

This project uses Prisma ORM to interact with a Supabase PostgreSQL database.

## Setup

1. Install Prisma dependencies:
   ```bash
   pnpm add prisma @prisma/client
   ```

2. Initialize Prisma:
   ```bash
   npx prisma init
   ```

3. Configure the database connection in `.env`:
   ```
   DATABASE_URL="postgres://postgres.goakphmfcayylqvnigvv:your_password@aws-0-us-west-1.pooler.supabase.com:5432/postgres"
   ```

   Replace `your_password` with the actual Supabase database password.

4. Generate the Prisma client:
   ```bash
   npx prisma generate
   ```

## Database Schema

The database schema includes the following models:

### WebhookResult

```prisma
model WebhookResult {
  id           String   @id @default(dbgenerated("extensions.uuid_generate_v4()")) @db.Uuid
  prediction_id String   @unique
  results      Json
  created_at   DateTime? @default(now()) @db.Timestamptz

  @@map("webhook_results")
  @@index([prediction_id], name: "idx_webhook_results_prediction_id")
}
```

### Account

```prisma
model Account {
  userId                 String  @id @unique
  categorisationRange    String?
  categorisationTab      String?
  columnOrderCategorisation Json?
  api_key                String? @unique

  @@map("account")
}
```

## Usage

Import the Prisma client in your code:

```typescript
import { prisma } from '@/lib/prisma';
```

Example usage:

```typescript
// Get an account by userId
const account = await prisma.account.findUnique({
  where: { userId: 'user123' }
});

// Create a webhook result
const webhookResult = await prisma.webhookResult.create({
  data: {
    prediction_id: 'pred123',
    results: { data: 'some data' }
  }
});
```

## API Routes

The following API routes use Prisma to interact with the database:

- `/api/account` - GET and POST endpoints for account operations
- `/api/webhook-results` - GET and POST endpoints for webhook result operations

## Migrations

To create and apply migrations:

```bash
npx prisma migrate dev --name init
```

To apply migrations in production:

```bash
npx prisma migrate deploy
``` 