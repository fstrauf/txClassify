const { Client } = require('pg');
require('dotenv').config();

async function main() {
  const client = new Client({
    host: process.env.PGHOST,
    port: 5432,
    user: process.env.PGUSER,
    password: process.env.PGPASSWORD,
    database: process.env.PGDATABASE,
    ssl: {
      rejectUnauthorized: true
    }
  });

  try {
    console.log('Connecting to Neon database...');
    await client.connect();
    console.log('Connection successful!');
    
    const result = await client.query('SELECT 1 as test_connection');
    console.log('Query result:', result.rows);
  } catch (error) {
    console.error('Connection failed:', error);
  } finally {
    await client.end();
  }
}

main(); 