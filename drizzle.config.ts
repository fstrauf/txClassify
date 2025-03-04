import type { Config } from 'drizzle-kit';
import * as dotenv from 'dotenv';

// Load environment variables
dotenv.config({ path: '.env.local' });

// Drizzle configuration
export default {
  schema: './db/schema',
  out: './db/migrations',
  dialect: 'postgresql',
  dbCredentials: {
    host: process.env.SUPABASE_DB_HOST || '',
    port: parseInt(process.env.SUPABASE_DB_PORT || '5432'),
    user: process.env.SUPABASE_DB_USER || '',
    password: process.env.SUPABASE_DB_PASSWORD || '',
    database: process.env.SUPABASE_DB_NAME || '',
    ssl: true,
  },
  // Verbose output for debugging
  verbose: true,
  // Strict mode for better type safety
  strict: true,
} satisfies Config; 