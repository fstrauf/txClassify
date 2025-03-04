import { pgTable, text, jsonb } from 'drizzle-orm/pg-core';

/**
 * Account table schema
 * 
 * This table stores user account information and configuration settings
 * for categorization tasks.
 */
export const account = pgTable('account', {
  // Auth0 user ID, used as the primary key
  userId: text('userId').notNull().primaryKey(),
  
  // The range that is to be selected for categorizing
  categorisationRange: text('categorisationRange'),
  
  // The tab of the sheet used for categorisation
  categorisationTab: text('categorisationTab'),
  
  // Order of categorisation columns in a user's sheet
  columnOrderCategorisation: jsonb('columnOrderCategorisation'),
  
  // API key for authentication with unique constraint
  apiKey: text('api_key').unique(),
}); 