import { drizzle } from 'drizzle-orm/postgres-js';
import { migrate } from 'drizzle-orm/postgres-js/migrator';
import postgres from 'postgres';
import * as dotenv from 'dotenv';

// Load environment variables
dotenv.config({ path: '.env.local' });

/**
 * Migration runner script
 * 
 * This script applies database migrations using Drizzle ORM.
 * It connects to the database using the credentials from environment variables.
 */

async function main() {
  console.log('Starting database migration...');
  
  try {
    // Get database connection details from environment variables
    const connectionString = process.env.DATABASE_URL || 
      `postgres://${process.env.SUPABASE_DB_USER}:${process.env.SUPABASE_DB_PASSWORD}@${process.env.SUPABASE_DB_HOST}:${process.env.SUPABASE_DB_PORT}/${process.env.SUPABASE_DB_NAME}`;
    
    // Create a Postgres client for migrations
    const migrationClient = postgres(connectionString, { max: 1, ssl: 'require' });
    
    // Initialize Drizzle with the migration client
    const db = drizzle(migrationClient);
    
    // Run migrations
    console.log('Applying migrations...');
    await migrate(db, { migrationsFolder: './db/migrations' });
    
    console.log('Migrations completed successfully!');
    
    // Close the connection
    await migrationClient.end();
    process.exit(0);
  } catch (error) {
    console.error('Migration failed:', error);
    process.exit(1);
  }
}

// Run the migration
main(); 