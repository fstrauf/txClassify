const fs = require("fs");
const { execSync } = require("child_process");
const path = require("path");

// Determine the environment
const isVercel = process.env.VERCEL === "1";
const isRender = process.env.RENDER === "1";

// Store original working directory
const originalDir = process.cwd();

try {
  // Path to the original schema
  const originalSchemaPath = path.join(__dirname, "../prisma/schema.prisma");
  const originalSchema = fs.readFileSync(originalSchemaPath, "utf8");

  if (isVercel) {
    console.log("Generating Prisma client for Vercel...");
    // For Vercel, remove the Python client generator
    const vercelSchema = originalSchema.replace(/generator python_client {[\s\S]*?}\n\n/, "");
    fs.writeFileSync(originalSchemaPath, vercelSchema);

    // Generate only TypeScript client
    execSync("npx prisma generate", { stdio: "inherit" });
    console.log("Successfully generated TypeScript client for Vercel");
  } else if (isRender) {
    console.log("Generating Prisma client for Render...");
    // For Render, ensure pythonHandler/prisma directory exists
    const pythonPrismaDir = path.join(__dirname, "../pythonHandler/prisma");
    if (!fs.existsSync(pythonPrismaDir)) {
      fs.mkdirSync(pythonPrismaDir, { recursive: true });
    }

    // Copy schema to Python location and update output path
    let pythonSchema = originalSchema;
    pythonSchema = pythonSchema.replace(/output(\s*)=(\s*)"[^"]*"/, 'output = "../client"');

    fs.writeFileSync(path.join(pythonPrismaDir, "schema.prisma"), pythonSchema);

    // Generate from Python location
    process.chdir(pythonPrismaDir);
    execSync("npx prisma generate", { stdio: "inherit" });
    console.log("Successfully generated Prisma clients for Render");
  } else {
    console.log("Generating Prisma clients for local development...");
    // Local development - generate both
    execSync("npx prisma generate", { stdio: "inherit" });

    // Also generate Python client
    const pythonPrismaDir = path.join(__dirname, "../pythonHandler/prisma");
    if (!fs.existsSync(pythonPrismaDir)) {
      fs.mkdirSync(pythonPrismaDir, { recursive: true });
    }

    let pythonSchema = originalSchema;
    pythonSchema = pythonSchema.replace(/output(\s*)=(\s*)"[^"]*"/, 'output = "../client"');

    fs.writeFileSync(path.join(pythonPrismaDir, "schema.prisma"), pythonSchema);

    process.chdir(pythonPrismaDir);
    execSync("npx prisma generate", { stdio: "inherit" });
    console.log("Successfully generated both Prisma clients for local development");
  }
} catch (error) {
  console.error("Error generating Prisma clients:", error);
  process.exit(1);
} finally {
  // Restore original working directory
  process.chdir(originalDir);
}
