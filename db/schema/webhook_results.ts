import { pgTable, text, jsonb, uuid, timestamp, index } from 'drizzle-orm/pg-core';

/**
 * Webhook Results table schema
 * 
 * This table stores results from webhook calls, particularly for prediction results.
 */
export const webhookResults = pgTable('webhook_results', {
  // Unique identifier for the webhook result
  id: uuid('id').defaultRandom().primaryKey(),
  
  // ID of the prediction this webhook result is associated with
  predictionId: text('prediction_id').notNull().unique(),
  
  // JSON data containing the results
  results: jsonb('results').notNull(),
  
  // Timestamp when the record was created
  createdAt: timestamp('created_at', { withTimezone: true }).defaultNow(),
}, (table) => {
  return {
    // Index on prediction_id for faster lookups
    predictionIdIdx: index('idx_webhook_results_prediction_id').on(table.predictionId),
  };
}); 