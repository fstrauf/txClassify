// test-utils.js - Utility functions extracted from test-api-endpoints.js
const fs = require('fs');
const path = require('path');
const csv = require('csv-parser');

const loadCategorizationData = (file_name) => {
  return new Promise((resolve, reject) => {
    const transactions = [];
    const targetCsvPath = path.join(__dirname, "test_data", file_name);
    let headersProcessed = false;
    let descriptionIndex = -1;
    let amountIndex = -1;
    let codeIndex = -1;

    console.log(`Loading categorization data from ${file_name}...`);

    const stream = fs
      .createReadStream(targetCsvPath)
      .pipe(
        csv({
          mapHeaders: ({ header, index }) => {
            const lowerHeader = header.toLowerCase().trim();
            if (["description", "narrative", "details"].includes(lowerHeader)) {
              return "description";
            }
            if (["amount", "amount spent"].includes(lowerHeader)) {
              return "amount";
            }
            if (["code"].includes(lowerHeader)) {
              return "code";
            }
            return null;
          },
        })
      )
      .on("headers", (headers) => {
        descriptionIndex = headers.indexOf("description");
        amountIndex = headers.indexOf("amount");
        codeIndex = headers.indexOf("code");
        headersProcessed = true;

        console.log(`Header indices found: Description=${descriptionIndex}, Amount=${amountIndex}, Code=${codeIndex}`);

        if (descriptionIndex === -1) {
          stream.destroy();
          return reject(
            new Error("Categorization CSV must contain a header named 'Description', 'Narrative', or 'Details'")
          );
        }
        if (amountIndex === -1) {
          console.log("Optional 'Amount' or 'Amount Spent' header not found. Proceeding without amount/money_in data.");
        }
        if (codeIndex === -1) {
          console.log("Optional 'Code' header not found. Proceeding without code data for combining descriptions.");
        }
      })
      .on("data", (row) => {
        if (!headersProcessed) return;

        const description = row.description;
        const amountValue = amountIndex !== -1 ? row.amount : undefined;
        const codeValue = codeIndex !== -1 ? row.code : undefined;

        if (description) {
          let parsedAmount = null;
          let money_in = null;

          if (amountIndex !== -1 && amountValue !== undefined && amountValue !== null) {
            const cleanAmount = String(amountValue).replace(/[^\d.-]/g, "");
            parsedAmount = parseFloat(cleanAmount);
            if (!isNaN(parsedAmount)) {
              money_in = parsedAmount >= 0;
            } else {
              console.log(`Could not parse amount: "${amountValue}" for description: "${description}"`);
              parsedAmount = null;
            }
          }

          transactions.push({
            description: description,
            money_in: money_in,
            amount: parsedAmount,
            code: codeValue,
          });
        }
      })
      .on("end", () => {
        const transactionObjects = transactions.filter((t) => typeof t === "object");
        console.log(`Loaded ${transactions.length} descriptions from ${file_name}`);
        console.log(
          `${transactionObjects.filter((t) => t.money_in !== null).length} of ${
            transactions.length
          } have money_in flag set`
        );
        console.log(
          `${transactionObjects.filter((t) => t.amount !== null).length} of ${
            transactions.length
          } have amount field set`
        );
        console.log(
          `${transactionObjects.filter((t) => t.code !== undefined).length} of ${
            transactions.length
          } have code field set`
        );

        if (transactions.length > 0) {
          console.log("Sample transactions (first 3):");
          for (let i = 0; i < Math.min(3, transactions.length); i++) {
            console.log(`  ${i + 1}: ${JSON.stringify(transactions[i])}`);
          }
        }
        resolve(transactions);
      })
      .on("error", (error) => {
        console.error(`Error loading ${file_name}: ${error.message}`);
        reject(error);
      });
  });
};

module.exports = {
  loadCategorizationData
};
