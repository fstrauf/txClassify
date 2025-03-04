const { PrismaClient } = require('@prisma/client');

async function main() {
  console.log('Creating Prisma client...');
  const prisma = new PrismaClient({
    log: ['query', 'info', 'warn', 'error'],
  });

  try {
    console.log('Testing connection...');
    // Try a simple query that doesn't require existing tables
    const result = await prisma.$executeRaw`SELECT 1 as result`;
    console.log('Connection successful:', result);
  } catch (error) {
    console.error('Connection failed:', error);
    console.error('Error details:', error.message);
  } finally {
    await prisma.$disconnect();
    console.log('Disconnected from database');
  }
}

main(); 