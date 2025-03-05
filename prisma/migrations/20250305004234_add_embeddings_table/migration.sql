-- CreateTable
CREATE TABLE "embeddings" (
    "id" UUID NOT NULL DEFAULT gen_random_uuid(),
    "file_name" TEXT NOT NULL,
    "data" BYTEA NOT NULL,
    "created_at" TIMESTAMPTZ(6) DEFAULT CURRENT_TIMESTAMP,
    "updated_at" TIMESTAMPTZ(6),

    CONSTRAINT "embeddings_pkey" PRIMARY KEY ("id")
);

-- CreateIndex
CREATE UNIQUE INDEX "embeddings_file_name_key" ON "embeddings"("file_name");

-- CreateIndex
CREATE INDEX "idx_embedding_file_name" ON "embeddings"("file_name");
