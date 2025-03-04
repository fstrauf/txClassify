# Neon Database Migration Guide

This document outlines the steps taken to migrate the application from Supabase to Neon serverless PostgreSQL, along with recommendations for the Next.js application.

## Migration Summary

1. Updated connection strings in `.env` file to use Neon serverless PostgreSQL
2. Updated Prisma client configuration for better connection handling
3. Added error handling and retry logic for database operations
4. Updated API routes with improved error handling
5. Created test scripts to verify database operations

## Connection Configuration

The following environment variables are used for Neon database connection:

```
# Neon Serverless PostgreSQL Connection Strings
DATABASE_URL=postgres://neondb_owner:password@ep-autumn-lab-a4g3jyo8-pooler.us-east-1.aws.neon.tech:5432/neondb?sslmode=require&connect_timeout=30
DIRECT_URL=postgres://neondb_owner:password@ep-autumn-lab-a4g3jyo8.us-east-1.aws.neon.tech:5432/neondb?sslmode=require&connect_timeout=30

# Additional Neon connection parameters
PGHOST=ep-autumn-lab-a4g3jyo8-pooler.us-east-1.aws.neon.tech
PGHOST_UNPOOLED=ep-autumn-lab-a4g3jyo8.us-east-1.aws.neon.tech
PGUSER=neondb_owner
PGDATABASE=neondb
PGPASSWORD=password
```

## Prisma Configuration

The Prisma schema has been updated to include the `relationMode = "foreignKeys"` setting for better compatibility with Neon:

```prisma
datasource db {
  provider  = "postgresql"
  url       = env("DATABASE_URL")
  directUrl = env("DIRECT_URL")
  relationMode = "foreignKeys"
}
```

## Optimized Prisma Client

The Prisma client has been optimized for serverless environments:

```typescript
import { PrismaClient } from '@prisma/client';

declare global {
  var prisma: PrismaClient | undefined;
}

const prismaClientSingleton = () => {
  return new PrismaClient({
    log: ['error', 'warn'],
  });
};

export const prisma = globalThis.prisma || prismaClientSingleton();

if (process.env.NODE_ENV !== 'production') {
  globalThis.prisma = prisma;
}

// Ensure connections are properly closed
process.on('beforeExit', async () => {
  await prisma.$disconnect();
});

// Handle unexpected errors
process.on('uncaughtException', async (e) => {
  console.error('Uncaught exception:', e);
  await prisma.$disconnect();
  process.exit(1);
});

// Handle unhandled promise rejections
process.on('unhandledRejection', async (e) => {
  console.error('Unhandled rejection:', e);
  await prisma.$disconnect();
});

export default prisma;
```

## API Routes with Improved Error Handling

API routes have been updated with improved error handling for Prisma operations:

```typescript
import { NextRequest, NextResponse } from 'next/server';
import { prisma } from '@/lib/prisma';
import { Prisma } from '@prisma/client';

export async function GET(request: NextRequest) {
  try {
    // Database operations
  } catch (error) {
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
    
    return NextResponse.json({ error: 'Failed to process request' }, { status: 500 });
  }
}
```

## Recommendations for Serverless Environments

1. **Connection Pooling**: Neon provides connection pooling through the `-pooler` endpoint. Always use this endpoint for better performance.

2. **Retry Logic**: Implement retry logic with exponential backoff for database operations to handle temporary connection issues.

3. **Error Handling**: Add specific error handling for Prisma errors to provide better feedback to users.

4. **Connection Cleanup**: Ensure connections are properly closed to avoid connection leaks.

5. **Monitoring**: Monitor database performance and connection usage to identify potential issues.

6. **Timeouts**: Set appropriate timeouts for database operations to avoid hanging requests.

7. **Caching**: Consider implementing caching for frequently accessed data to reduce database load.

## Testing

Use the provided test scripts to verify database operations:

- `scripts/test-neon-prisma.js`: Tests both direct PostgreSQL and Prisma connections
- `scripts/test-api-endpoints.js`: Tests Prisma operations with retry logic
- `scripts/check-accounts.js`: Checks account records
- `scripts/check-webhook-results.js`: Checks webhook result records

## Python Backend

The Python backend has been updated to use Prisma for database operations. The following files have been updated:

- `pythonHandler/utils/prisma_client.py`: Prisma client for Python
- `pythonHandler/utils/db.py`: Database utility module for direct PostgreSQL connection

## Troubleshooting

If you encounter connection issues:

1. Check that the Neon database is running and accessible
2. Verify your connection strings in the `.env` file
3. Ensure your IP is allowed in Neon's connection settings
4. Try increasing connection timeout values
5. Check for connection pool exhaustion
6. Verify that you're using the correct endpoint (pooler vs. unpooled)
7. Check for SSL/TLS issues 