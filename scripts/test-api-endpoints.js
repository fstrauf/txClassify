const { PrismaClient } = require('@prisma/client');
require('dotenv').config();

// Helper function to retry operations
async function withRetry(operation, maxRetries = 3, delay = 1000) {
  let lastError;
  
  for (let attempt = 1; attempt <= maxRetries; attempt++) {
    try {
      return await operation();
    } catch (error) {
      console.warn(`Attempt ${attempt}/${maxRetries} failed: ${error.message}`);
      lastError = error;
      
      if (attempt < maxRetries) {
        console.log(`Retrying in ${delay}ms...`);
        await new Promise(resolve => setTimeout(resolve, delay));
        // Increase delay for next attempt (exponential backoff)
        delay *= 2;
      }
    }
  }
  
  throw lastError;
}

// Test account operations
async function testAccountOperations(prisma) {
  console.log('\n=== Testing Account Operations ===');
  
  // Test userId
  const testUserId = 'test-user-' + Date.now();
  const testApiKey = 'test-api-key-' + Date.now();
  
  try {
    // Create account
    console.log(`Creating account for user ${testUserId}...`);
    const createdAccount = await withRetry(() => prisma.account.upsert({
      where: { userId: testUserId },
      update: { api_key: testApiKey },
      create: {
        userId: testUserId,
        api_key: testApiKey,
        categorisationRange: null,
        categorisationTab: null,
        columnOrderCategorisation: null
      }
    }));
    
    console.log('✅ Account created successfully:', createdAccount);
    
    // Get account
    console.log(`\nFetching account for user ${testUserId}...`);
    const fetchedAccount = await withRetry(() => prisma.account.findUnique({
      where: { userId: testUserId }
    }));
    
    console.log('✅ Account fetched successfully:', fetchedAccount);
    
    // Verify API key matches
    if (fetchedAccount.api_key === testApiKey) {
      console.log('✅ API key verification successful');
    } else {
      console.error('❌ API key verification failed');
    }
    
    return true;
  } catch (error) {
    console.error('❌ Account operations test failed:', error.message);
    return false;
  }
}

// Test webhook results operations
async function testWebhookResultsOperations(prisma) {
  console.log('\n=== Testing Webhook Results Operations ===');
  
  // Test prediction ID
  const testPredictionId = 'test-prediction-' + Date.now();
  const testResults = {
    status: 'success',
    confidence: 0.95,
    timestamp: new Date().toISOString(),
  };
  
  try {
    // Create webhook result
    console.log(`Creating webhook result for prediction ${testPredictionId}...`);
    const createdResult = await withRetry(() => prisma.webhookResult.create({
      data: {
        prediction_id: testPredictionId,
        results: testResults
      }
    }));
    
    console.log('✅ Webhook result created successfully:', {
      id: createdResult.id,
      prediction_id: createdResult.prediction_id
    });
    
    // Get webhook result
    console.log(`\nFetching webhook result for prediction ${testPredictionId}...`);
    const fetchedResult = await withRetry(() => prisma.webhookResult.findUnique({
      where: { prediction_id: testPredictionId }
    }));
    
    console.log('✅ Webhook result fetched successfully:', {
      id: fetchedResult.id,
      prediction_id: fetchedResult.prediction_id
    });
    
    // Verify prediction ID matches
    if (fetchedResult.prediction_id === testPredictionId) {
      console.log('✅ Prediction ID verification successful');
    } else {
      console.error('❌ Prediction ID verification failed');
    }
    
    return true;
  } catch (error) {
    console.error('❌ Webhook results operations test failed:', error.message);
    return false;
  }
}

// Main function to run all tests
async function main() {
  console.log('🔍 Starting Prisma operations tests');
  console.log('DATABASE_URL:', process.env.DATABASE_URL ? '✓ Set' : '✗ Not set');
  
  // Create Prisma client
  const prisma = new PrismaClient({
    log: ['error', 'warn'],
  });
  
  try {
    // Test connection
    console.log('Testing database connection...');
    const result = await withRetry(() => prisma.$queryRaw`SELECT 1 as result`);
    console.log('✅ Database connection successful:', result);
    
    // Test account operations
    const accountSuccess = await testAccountOperations(prisma);
    
    // Test webhook results operations
    const webhookSuccess = await testWebhookResultsOperations(prisma);
    
    // Summary
    console.log('\n=== Test Summary ===');
    console.log(`Account Operations: ${accountSuccess ? '✅ SUCCESS' : '❌ FAILED'}`);
    console.log(`Webhook Results Operations: ${webhookSuccess ? '✅ SUCCESS' : '❌ FAILED'}`);
    
    if (!accountSuccess || !webhookSuccess) {
      console.log('\n❗ Some tests failed. Please check the logs for details.');
      process.exit(1);
    }
  } catch (error) {
    console.error('❌ Database connection failed:', error.message);
    process.exit(1);
  } finally {
    // Disconnect Prisma client
    await prisma.$disconnect();
    console.log('Disconnected from database');
  }
}

// Run the tests
main()
  .catch(e => {
    console.error('Unexpected error:', e);
    process.exit(1);
  }); 