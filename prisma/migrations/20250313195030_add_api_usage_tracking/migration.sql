-- AlterTable
ALTER TABLE "account"
ADD COLUMN "lastUsed" TIMESTAMPTZ(6),
    ADD COLUMN "requestsCount" INTEGER NOT NULL DEFAULT 0;
-- AlterTable
ALTER TABLE "embeddings"
ADD COLUMN "accountId" TEXT;
-- CreateIndex
CREATE INDEX "idx_embedding_account_id" ON "embeddings"("accountId");
-- AddForeignKey
ALTER TABLE "embeddings"
ADD CONSTRAINT "embeddings_accountId_fkey" FOREIGN KEY ("accountId") REFERENCES "account"("userId") ON DELETE
SET NULL ON UPDATE CASCADE;