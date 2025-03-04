// This script generates a Prisma schema based on the existing tables
const fs = require('fs');
const path = require('path');

const schema = `// This is your Prisma schema file,
// learn more about it in the docs: https://pris.ly/d/prisma-schema

generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

model WebhookResult {
  id           String   @id @default(dbgenerated("extensions.uuid_generate_v4()")) @db.Uuid
  prediction_id String   @unique
  results      Json
  created_at   DateTime? @default(now()) @db.Timestamptz

  @@map("webhook_results")
  @@index([prediction_id], name: "idx_webhook_results_prediction_id")
}

model Account {
  userId                 String  @id @unique
  categorisationRange    String?
  categorisationTab      String?
  columnOrderCategorisation Json?
  api_key                String? @unique

  @@map("account")
}
`;

// Ensure the prisma directory exists
const prismaDir = path.join(__dirname, '..', 'prisma');
if (!fs.existsSync(prismaDir)) {
  fs.mkdirSync(prismaDir);
}

// Write the schema to the file
fs.writeFileSync(path.join(prismaDir, 'schema.prisma'), schema);

console.log('Prisma schema generated successfully!'); 