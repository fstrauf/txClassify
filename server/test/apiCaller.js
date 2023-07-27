const Replicate = require("replicate");
const fs = require("fs");
const csv = require("csv-parser");

const replicate = new Replicate({
  auth: process.env.REPLICATE_API_TOKEN,
});

async function runModel() {
  let text_batch = await new Promise((resolve, reject) => {
    let data = [];
    fs.createReadStream("new_data.csv")
      // fs.createReadStream("trained_data.csv")
      .pipe(csv({ headers: false }))
      .on("data", (row) => {
        // Get the keys of the row (column names)
        const keys = Object.keys(row);
        // Push the value of the second column (index 1) into the array
        data.push(row[keys[1]]);
      })
      .on("end", () => {
        resolve(data);
      })
      .on("error", reject);
  });

  const prediction = await replicate.predictions.create({
    version: "b6b7585c9640cd7a9572c6e129c9549d79c9c31f0d3fdce7baac7c67ca38f305",
    input: {
      text_batch: JSON.stringify(text_batch),
    },
    // webhook: "https://pythonhandler-yxxxtrqkpa-ts.a.run.app",
    // webhook: "https://65d5-206-83-122-86.ngrok-free.app/saveTrainedData",
    webhook: "https://65d5-206-83-122-86.ngrok-free.app/classify",
    webhook_events_filter: ["completed"],
  });

  console.log(prediction);
}

runModel();
