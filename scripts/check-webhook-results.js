const { PrismaClient } = require('@prisma/client');

const prisma = new PrismaClient();

async function main() {
  try {
    console.log('Checking webhook_results records...');
    
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
    }
  } catch (error) {
    console.error('Error checking webhook results:', error);
  } finally {
    await prisma.$disconnect();
  }
}

main()
  .catch(e => {
    console.error(e);
    process.exit(1);
  }); 