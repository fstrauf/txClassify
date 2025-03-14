-- CreateExtension
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- CreateTable
CREATE TABLE "webhook_results" (
    "id" UUID NOT NULL DEFAULT gen_random_uuid(),
    "prediction_id" TEXT NOT NULL,
    "results" JSONB NOT NULL,
    "created_at" TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,

    CONSTRAINT "webhook_results_pkey" PRIMARY KEY ("id")
);

-- CreateTable
CREATE TABLE "account" (
    "userId" TEXT NOT NULL,
    "categorisationRange" TEXT,
    "categorisationTab" TEXT,
    "columnOrderCategorisation" JSONB,
    "api_key" TEXT,

    CONSTRAINT "account_pkey" PRIMARY KEY ("userId")
);

-- CreateIndex
CREATE UNIQUE INDEX "webhook_results_prediction_id_key" ON "webhook_results"("prediction_id");

-- CreateIndex
CREATE INDEX "idx_webhook_results_prediction_id" ON "webhook_results"("prediction_id");

-- CreateIndex
CREATE UNIQUE INDEX "account_userId_key" ON "account"("userId");

-- CreateIndex
CREATE UNIQUE INDEX "account_api_key_key" ON "account"("api_key");
