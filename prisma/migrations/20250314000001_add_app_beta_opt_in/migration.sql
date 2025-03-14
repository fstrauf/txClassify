-- Create the enum type
CREATE TYPE "AppBetaOptInStatus" AS ENUM ('OPTED_IN', 'DISMISSED');
-- Add the column to the account table
ALTER TABLE "account"
ADD COLUMN "appBetaOptIn" "AppBetaOptInStatus";