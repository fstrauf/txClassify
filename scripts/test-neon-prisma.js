const { PrismaClient } = require('@prisma/client');
const { Client } = require('pg');
require('dotenv').config();

async function testDirectConnection() {
  console.log('\n=== Testing Direct PostgreSQL Connection ===');
  const client = new Client({
    host: process.env.PGHOST,
    port: 5432,
    user: process.env.PGUSER,
    password: process.env.PGPASSWORD,
    database: process.env.PGDATABASE,
    ssl: {
      rejectUnauthorized: true
    },
    // Add connection timeout
    connectionTimeoutMillis: 10000,
  });

  try {
    console.log('Connecting to Neon database directly...');
    await client.connect();
    console.log('✅ Direct connection successful!');
    
    const result = await client.query('SELECT 1 as test_connection');
    console.log('Query result:', result.rows);
    
    return true;
  } catch (error) {
    console.error('❌ Direct connection failed:', error.message);
    return false;
  } finally {
    await client.end();
    console.log('Direct connection closed');
  }
}

async function testPrismaConnection() {
  console.log('\n=== Testing Prisma Connection ===');
  console.log('Creating Prisma client...');
  
  const prisma = new PrismaClient({
    log: ['query', 'error'],
    datasources: {
      db: {
        url: process.env.DATABASE_URL,
      },
    },
  });

  try {
    console.log('Testing Prisma connection...');
    // Try a simple query that doesn't require existing tables
    const result = await prisma.$queryRaw`SELECT 1 as result`;
    console.log('✅ Prisma connection successful:', result);
    
    return true;
  } catch (error) {
    console.error('❌ Prisma connection failed:', error.message);
    if (error.meta) {
      console.error('Error details:', error.meta);
    }
    return false;
  } finally {
    await prisma.$disconnect();
    console.log('Prisma connection closed');
  }
}

async function checkAccounts(prisma) {
  console.log('\n=== Checking Account Records ===');
  try {
    // Count the number of accounts
    const accountCount = await prisma.account.count();
    console.log(`Found ${accountCount} account records in the database`);
    
    // Get a sample of accounts (first 3)
    if (accountCount > 0) {
      const sampleAccounts = await prisma.account.findMany({
        take: 3,
      });
      
      console.log('Sample accounts:');
      sampleAccounts.forEach(account => {
        console.log(`- User ID: ${account.userId}`);
        console.log(`  API Key: ${account.api_key ? 'Set' : 'Not set'}`);
        console.log('---');
      });
      return true;
    }
    return accountCount > 0;
  } catch (error) {
    console.error('Error checking accounts:', error.message);
    return false;
  }
}

async function checkWebhookResults(prisma) {
  console.log('\n=== Checking Webhook Results ===');
  try {
    // Count the number of webhook results
    const count = await prisma.webhookResult.count();
    console.log(`Found ${count} webhook_results records in the database`);
    
    // Get a sample of webhook results (first 3)
    if (count > 0) {
      const samples = await prisma.webhookResult.findMany({
        take: 3,
      });
      
      console.log('Sample webhook results:');
      samples.forEach(result => {
        console.log(`- ID: ${result.id}`);
        console.log(`  Prediction ID: ${result.prediction_id}`);
        console.log(`  Created At: ${result.created_at}`);
        console.log('---');
      });
      return true;
    }
    return count > 0;
  } catch (error) {
    console.error('Error checking webhook results:', error.message);
    return false;
  }
}

async function main() {
  console.log('🔍 Starting comprehensive Neon database connection test');
  console.log('DATABASE_URL:', process.env.DATABASE_URL ? '✓ Set' : '✗ Not set');
  console.log('DIRECT_URL:', process.env.DIRECT_URL ? '✓ Set' : '✗ Not set');
  
  // Test direct PostgreSQL connection
  const directConnSuccess = await testDirectConnection();
  
  // Test Prisma connection
  const prismaConnSuccess = await testPrismaConnection();
  
  // If Prisma connection was successful, check data
  if (prismaConnSuccess) {
    const prisma = new PrismaClient();
    try {
      await checkAccounts(prisma);
      await checkWebhookResults(prisma);
    } finally {
      await prisma.$disconnect();
    }
  }
  
  // Summary
  console.log('\n=== Test Summary ===');
  console.log(`Direct PostgreSQL Connection: ${directConnSuccess ? '✅ SUCCESS' : '❌ FAILED'}`);
  console.log(`Prisma Connection: ${prismaConnSuccess ? '✅ SUCCESS' : '❌ FAILED'}`);
  
  if (!directConnSuccess && !prismaConnSuccess) {
    console.log('\n❗ Both connection methods failed. Please check your connection settings.');
    console.log('Suggestions:');
    console.log('1. Verify your Neon database is running and accessible');
    console.log('2. Check your connection strings in .env file');
    console.log('3. Ensure your IP is allowed in Neon\'s connection settings');
    console.log('4. Try increasing connection timeout values');
  }
}

main()
  .catch(e => {
    console.error('Unexpected error:', e);
    process.exit(1);
  }); 