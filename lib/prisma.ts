import { PrismaClient } from '@prisma/client';

// Define the global type for PrismaClient
declare global {
  var prisma: PrismaClient | undefined;
}

// PrismaClient is attached to the `global` object in development to prevent
// exhausting your database connection limit.
// Learn more: https://pris.ly/d/help/next-js-best-practices

// Connection management options for serverless environments
const prismaClientSingleton = () => {
  return new PrismaClient({
    log: ['error', 'warn'],
  });
};

// Use existing client instance if available to avoid connection limit issues
export const prisma = globalThis.prisma || prismaClientSingleton();

// In development, attach to global to prevent multiple instances
if (process.env.NODE_ENV !== 'production') {
  globalThis.prisma = prisma;
}

// Ensure connections are properly closed before the process exits
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