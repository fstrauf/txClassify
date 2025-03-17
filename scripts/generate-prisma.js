const { execSync } = require("child_process");

// Determine the environment
const isVercel = process.env.VERCEL === "1";
const isRender = process.env.RENDER === "1";

try {
  if (isVercel) {
    // For Vercel (Next.js frontend), we only need the TypeScript client
    console.log("Generating TypeScript Prisma client for Vercel...");
    execSync("npx prisma@5.17.0 generate", { stdio: "inherit" });
  } else if (isRender) {
    // For Render (Python backend), we only need the Python client
    console.log("Generating Python Prisma client for Render...");
    execSync("python3 -m pip install prisma==5.17.0", { stdio: "inherit" });
    execSync("python3 -m prisma generate", { stdio: "inherit" });
  } else {
    // For local development, generate the TypeScript client
    // If you need Python client locally, run `python3 -m prisma generate` manually
    console.log("Generating TypeScript Prisma client for local development...");
    execSync("npx prisma@5.17.0 generate", { stdio: "inherit" });
  }
} catch (error) {
  console.error("Error generating Prisma client:", error);
  process.exit(1);
}
