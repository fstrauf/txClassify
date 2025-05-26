// Test script specifically for embedding-based grouping functionality
const path = require("path");
require("dotenv").config({ path: path.resolve(__dirname, "../.env") });

const TEST_API_KEY = process.env.TEST_API_KEY;
const API_URL = process.env.TARGET_API_URL || "http://localhost";

const log = (message) => {
  const timestamp = new Date().toISOString();
  console.log(`[${timestamp}] ${message}`);
};

const testEmbeddingGrouping = async () => {
  log("=== Testing Embedding-Based Grouping ===");
  
  // Test data with merchants that should be grouped together
  const testDescriptions = [
    "4835-****-****-0311  Df Woolworths N",
    "4835-****-****-0329  Df Woolworths N", 
    "4835-****-****-0311  Df Woolworths O",
    "4835-****-****-0329  Df New World Mt",
    "4835-****-****-0311  Df New World Mt",
    "4835-****-****-0329  Df New World Bl",
    "4835-****-****-0311  Df Kmart - Bayf",
    "4835-****-****-0329  Df Kmart",
    "4835-****-****-0329  Df Kmart - Bayf",
    "4835-****-****-0311  Df The Warehous",
    "4835-****-****-0329  Df The Warehous"
  ];

  try {
    log("1. Testing regular cleaning without grouping...");
    const regularResponse = await fetch(`${API_URL}/clean_text`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": TEST_API_KEY,
      },
      body: JSON.stringify({ 
        descriptions: testDescriptions 
      }),
    });

    if (!regularResponse.ok) {
      throw new Error(`Regular cleaning failed: ${regularResponse.status}`);
    }

    const regularResult = await regularResponse.json();
    log("Regular cleaning results:");
    regularResult.cleaned_descriptions.forEach((desc, i) => {
      console.log(`  ${i + 1}: "${testDescriptions[i]}" -> "${desc}"`);
    });

    log("\n2. Testing embedding-based grouping with HDBSCAN...");
    const embeddingResponse = await fetch(`${API_URL}/clean_text`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        "X-API-Key": TEST_API_KEY,
      },
      body: JSON.stringify({ 
        descriptions: testDescriptions,
        use_embedding_grouping: true,
        embedding_clustering_method: "hdbscan",
        embedding_similarity_threshold: 0.8
      }),
    });

    if (!embeddingResponse.ok) {
      const errorText = await embeddingResponse.text();
      throw new Error(`Embedding grouping failed: ${embeddingResponse.status} - ${errorText}`);
    }

    const embeddingResult = await embeddingResponse.json();
    log("Embedding-based grouping results:");
    
    if (embeddingResult.groups) {
      log("Groups found:");
      Object.entries(embeddingResult.groups).forEach(([groupName, members]) => {
        console.log(`  Group "${groupName}": ${members.length} members`);
        members.forEach(member => console.log(`    - "${member}"`));
      });
    } else {
      log("No groups returned in response");
    }

    log("Cleaned descriptions:");
    embeddingResult.cleaned_descriptions.forEach((desc, i) => {
      console.log(`  ${i + 1}: "${testDescriptions[i]}" -> "${desc}"`);
    });

    log("\n3. Testing with different clustering methods...");
    
    const methods = ["dbscan", "hierarchical", "similarity"];
    for (const method of methods) {
      log(`\nTesting with ${method}...`);
      
      const methodResponse = await fetch(`${API_URL}/clean_text`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "X-API-Key": TEST_API_KEY,
        },
        body: JSON.stringify({ 
          descriptions: testDescriptions.slice(0, 6), // Use fewer descriptions for faster testing
          use_embedding_grouping: true,
          embedding_clustering_method: method,
          embedding_similarity_threshold: 0.8
        }),
      });

      if (methodResponse.ok) {
        const methodResult = await methodResponse.json();
        if (methodResult.groups) {
          const groupCount = Object.keys(methodResult.groups).length;
          log(`  ${method}: Found ${groupCount} groups`);
        } else {
          log(`  ${method}: No groups returned`);
        }
      } else {
        log(`  ${method}: Failed with status ${methodResponse.status}`);
      }
    }

    log("\n=== Embedding-Based Grouping Test Completed ===");
    return true;

  } catch (error) {
    log(`Test failed: ${error.message}`);
    console.error(error);
    return false;
  }
};

// Run the test
testEmbeddingGrouping().then(success => {
  process.exit(success ? 0 : 1);
});
