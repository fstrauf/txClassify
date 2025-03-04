// Import the PrismaClient
const { PrismaClient } = require('@prisma/client');

// Create a new PrismaClient instance with logging enabled
const prisma = new PrismaClient({
  log: ['query', 'info', 'warn', 'error'],
  datasources: {
    db: {
      url: process.env.DATABASE_URL,
    },
  },
});

// Main function to test the connection
async function main() {
  console.log('Testing connection to Neon database...');
  
  try {
    // Test connection with a simple query
    const result = await prisma.$queryRaw`SELECT 1 as result`;
    console.log('Connection successful!');
    console.log('Query result:', result);
    
    // Test account table
    console.log('\nTesting account table...');
    const accounts = await prisma.account.findMany({
      take: 5,
    });
    console.log(`Found ${accounts.length} accounts`);
    if (accounts.length > 0) {
      console.log('First account:', accounts[0]);
    }
    
    // Test webhook_results table
    console.log('\nTesting webhook_results table...');
    const webhookResults = await prisma.webhookResult.findMany({
      take: 5,
    });
    console.log(`Found ${webhookResults.length} webhook results`);
    if (webhookResults.length > 0) {
      console.log('First webhook result ID:', webhookResults[0].id);
      console.log('First webhook result prediction_id:', webhookResults[0].prediction_id);
    }
    
    return { success: true };
  } catch (error) {
    console.error('Connection failed:', error);
    return { success: false, error };
  } finally {
    // Always disconnect the client
    await prisma.$disconnect();
    console.log('Disconnected from database');
  }
}

// Run the test
main()
  .then((result) => {
    console.log('\nTest completed:', result.success ? 'SUCCESS' : 'FAILED');
    if (!result.success) {
      console.error('Error details:', result.error);
      process.exit(1);
    }
  })
  .catch((e) => {
    console.error('Unexpected error:', e);
    process.exit(1);
  }); 