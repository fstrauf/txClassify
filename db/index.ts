import { drizzle } from 'drizzle-orm/postgres-js';
import postgres from 'postgres';
import * as schema from './schema';
import { PostgresJsDatabase } from 'drizzle-orm/postgres-js';

/**
 * Database client setup
 * 
 * This file sets up Drizzle ORM for server-side API routes.
 * Client-side components will use API routes to access the database.
 */

// For server environments (Node.js) only
// Get database connection details from environment variables
const connectionString = process.env.DATABASE_URL;

// Only initialize if we have a connection string and we're in a server context
let queryClient: ReturnType<typeof postgres> | undefined;
let db: PostgresJsDatabase<typeof schema> | undefined;

// Only initialize Drizzle in server context
if (typeof window === 'undefined' && connectionString) {
  try {
    queryClient = postgres(connectionString, { 
      max: 10, // Maximum number of connections
      ssl: { rejectUnauthorized: false }, // Enable SSL for secure connections with Supabase
      connect_timeout: 30, // Increase connection timeout to 30 seconds
      idle_timeout: 30, // Keep connections alive longer
    });
    
    // Initialize Drizzle with the query client and schema
    db = drizzle(queryClient, { schema });
    console.log('Drizzle ORM initialized successfully for server-side');
  } catch (error) {
    console.error('Failed to initialize Drizzle ORM:', error);
  }
}

// Export database client and schema
export { db, schema }; 