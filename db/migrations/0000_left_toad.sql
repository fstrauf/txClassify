CREATE TABLE "account" (
	"userId" text PRIMARY KEY NOT NULL,
	"categorisationRange" text,
	"categorisationTab" text,
	"columnOrderCategorisation" jsonb,
	"api_key" text,
	CONSTRAINT "account_api_key_unique" UNIQUE("api_key")
);
--> statement-breakpoint
CREATE TABLE "webhook_results" (
	"id" uuid PRIMARY KEY DEFAULT gen_random_uuid() NOT NULL,
	"prediction_id" text NOT NULL,
	"results" jsonb NOT NULL,
	"created_at" timestamp with time zone DEFAULT now(),
	CONSTRAINT "webhook_results_prediction_id_unique" UNIQUE("prediction_id")
);
--> statement-breakpoint
CREATE INDEX "idx_webhook_results_prediction_id" ON "webhook_results" USING btree ("prediction_id");