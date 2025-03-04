#!/usr/bin/env node

/**
 * This script helps set up the database environment variables for Drizzle migrations.
 * It extracts the database connection details from the Supabase connection string.
 */

const fs = require('fs');
const path = require('path');
const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

// Function to parse a PostgreSQL connection string
function parseConnectionString(connectionString) {
  try {
    // Remove postgres:// or postgresql:// prefix
    const cleanUrl = connectionString.replace(/^(postgres|postgresql):\/\//, '');
    
    // Split into credentials and host parts
    const [credentials, hostPart] = cleanUrl.split('@');
    const [username, password] = credentials.split(':');
    
    // Split host part into host:port/database
    const [hostAndPort, database] = hostPart.split('/');
    const [host, port] = hostAndPort.split(':');
    
    return {
      host,
      port: port || '5432',
      username,
      password,
      database
    };
  } catch (error) {
    console.error('Error parsing connection string:', error);
    return null;
  }
}

// Function to update .env.local file
function updateEnvFile(envPath, dbDetails) {
  try {
    // Read existing .env.local file if it exists
    let envContent = '';
    if (fs.existsSync(envPath)) {
      envContent = fs.readFileSync(envPath, 'utf8');
    }
    
    // Define the variables to add or update
    const dbVars = {
      'SUPABASE_DB_HOST': dbDetails.host,
      'SUPABASE_DB_PORT': dbDetails.port,
      'SUPABASE_DB_USER': dbDetails.username,
      'SUPABASE_DB_PASSWORD': dbDetails.password,
      'SUPABASE_DB_NAME': dbDetails.database,
      'DATABASE_URL': `postgres://${dbDetails.username}:${dbDetails.password}@${dbDetails.host}:${dbDetails.port}/${dbDetails.database}`
    };
    
    // Update or add each variable
    Object.entries(dbVars).forEach(([key, value]) => {
      // Check if the variable already exists
      const regex = new RegExp(`^${key}=.*$`, 'm');
      if (regex.test(envContent)) {
        // Update existing variable
        envContent = envContent.replace(regex, `${key}=${value}`);
      } else {
        // Add new variable
        envContent += `\n${key}=${value}`;
      }
    });
    
    // Write the updated content back to the file
    fs.writeFileSync(envPath, envContent);
    console.log(`Updated ${envPath} with database connection details.`);
  } catch (error) {
    console.error('Error updating .env.local file:', error);
  }
}

// Main function
function main() {
  console.log('Database Environment Setup');
  console.log('==========================');
  console.log('This script will help you set up the database environment variables for Drizzle migrations.');
  console.log('You will need your Supabase PostgreSQL connection string.');
  console.log('You can find this in your Supabase dashboard under Project Settings > Database > Connection String.');
  console.log('');
  
  rl.question('Enter your Supabase PostgreSQL connection string: ', (connectionString) => {
    const dbDetails = parseConnectionString(connectionString);
    
    if (!dbDetails) {
      console.error('Invalid connection string. Please try again.');
      rl.close();
      return;
    }
    
    console.log('\nParsed connection details:');
    console.log(`Host: ${dbDetails.host}`);
    console.log(`Port: ${dbDetails.port}`);
    console.log(`Username: ${dbDetails.username}`);
    console.log(`Password: ${'*'.repeat(dbDetails.password.length)}`);
    console.log(`Database: ${dbDetails.database}`);
    console.log('');
    
    rl.question('Do you want to update your .env.local file with these details? (y/n): ', (answer) => {
      if (answer.toLowerCase() === 'y') {
        const envPath = path.join(process.cwd(), '.env.local');
        updateEnvFile(envPath, dbDetails);
      }
      
      rl.close();
    });
  });
}

// Run the main function
main(); 