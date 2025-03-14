/*
 Warnings:
 
 - You are about to drop the column `file_name` on the `embeddings` table. All the data in the column will be lost.
 - A unique constraint covering the columns `[embedding_id]` on the table `embeddings` will be added. If there are existing duplicate values, this will fail.
 
 */
-- First add the new column allowing NULL values temporarily
ALTER TABLE "embeddings"
ADD COLUMN "embedding_id" TEXT;
-- Copy data from file_name to embedding_id
UPDATE "embeddings"
SET "embedding_id" = "file_name";
-- Now make the column NOT NULL after data is copied
ALTER TABLE "embeddings"
ALTER COLUMN "embedding_id"
SET NOT NULL;
-- DropIndex
DROP INDEX "embeddings_file_name_key";
-- DropIndex
DROP INDEX "idx_embedding_file_name";
-- Now drop the old column
ALTER TABLE "embeddings" DROP COLUMN "file_name";
-- CreateIndex
CREATE UNIQUE INDEX "embeddings_embedding_id_key" ON "embeddings"("embedding_id");
-- CreateIndex
CREATE INDEX "idx_embedding_embedding_id" ON "embeddings"("embedding_id");