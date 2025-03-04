const { PrismaClient } = require('@prisma/client');

const prisma = new PrismaClient({
  log: ['query', 'info', 'warn', 'error']
});

async function main() {
  try {
    console.log('Checking account records...');
    
    // Count the number of accounts
    const accountCount = await prisma.account.count();
    console.log(`Found ${accountCount} account records in the database`);
    
    // Get a sample of accounts (first 5)
    if (accountCount > 0) {
      const sampleAccounts = await prisma.account.findMany({
        take: 5,
      });
      
      console.log('Sample accounts:');
      sampleAccounts.forEach(account => {
        console.log(`- User ID: ${account.userId}`);
        console.log(`  Range: ${account.categorisationRange || 'N/A'}`);
        console.log(`  Tab: ${account.categorisationTab || 'N/A'}`);
        console.log(`  API Key: ${account.api_key ? 'Set' : 'Not set'}`);
        console.log('---');
      });
    }
  } catch (error) {
    console.error('Error checking accounts:', error);
  } finally {
    await prisma.$disconnect();
  }
}

main()
  .catch(e => {
    console.error(e);
    process.exit(1);
  }); 