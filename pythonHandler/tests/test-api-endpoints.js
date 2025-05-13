// Initial API call
const response = await axios.post(`${API_BASE_URL}/classify`, requestBody, {
  headers: {
    'Content-Type': 'application/json',
    Authorization: `Bearer ${apiKey}`,
  },
  validateStatus: function (status) {
    // Allow 200 and 202 without throwing errors
    return status >= 200 && status < 300;
  },
});

let results = null;
let predictionId = null;
let usedPolling = false;

// --- Handle initial response ---
if (response.status === 200) {
  // Could be synchronous completion OR asynchronous start
  if (response.data?.status === 'completed' && response.data?.results) {
    console.log('Categorization completed synchronously.');
    results = response.data.results;
  } else if (response.data?.status === 'processing' && response.data?.prediction_id) {
     console.log('Categorization started asynchronously (received 200 OK).');
     predictionId = response.data.prediction_id;
     usedPolling = true;
  } else {
     console.warn('Received 200 OK but response format unexpected:', response.data);
     throw new Error('Unexpected response format for status 200.');
  }
} else if (response.status === 202 && response.data?.prediction_id) {
  // Asynchronous start signaled by 202
  console.log('Categorization started asynchronously (received 202 Accepted).');
  predictionId = response.data.prediction_id;
  usedPolling = true;
} else {
  // Unexpected status or missing prediction_id for 202
  console.error('Error starting categorization:', response.status, response.data);
  throw new Error(`Error starting categorization: ${response.status}`);
}

// --- Polling if necessary ---
if (usedPolling && predictionId) {
  console.log(`Polling status for prediction ID: ${predictionId}...`);
  const maxRetries = 20; // e.g., 20 * 3 seconds = 60 seconds total timeout
  const pollInterval = 3000; // 3 seconds

  for (let i = 0; i < maxRetries; i++) {
    await new Promise((resolve) => setTimeout(resolve, pollInterval));

    try {
      const statusResponse = await axios.get(
        `${API_BASE_URL}/status/${predictionId}`,
        {
          headers: {
            Authorization: `Bearer ${apiKey}`,
          },
        }
      );

      const statusData = statusResponse.data;
      console.log(`Poll ${i + 1}: Status = ${statusData.status}`);

      if (statusData.status === 'completed' || statusData.status === 'succeeded') {
        console.log('Polling complete: Status is completed.');
        // Extract results from result_data
        if (statusData.result_data && typeof statusData.result_data === 'string') {
           try {
             const parsedResultData = JSON.parse(statusData.result_data);
             if (parsedResultData.results) {
                results = parsedResultData.results;
                console.log('Successfully parsed results from result_data.');
             } else {
                console.error('Completed, but \'results\' field missing within parsed result_data:', parsedResultData);
                throw new Error('Completed, but results data format is invalid.');
             }
           } catch (e) {
               console.error('Error parsing result_data JSON:', e);
               throw new Error('Completed, but failed to parse results data JSON.');
           }
        } else if (statusData.results) {
            // Fallback if results are somehow directly in the status response
            console.warn('Using results directly from status response (fallback).');
            results = statusData.results;
        } else {
           console.error('Completed, but result_data or results field missing in status response:', statusData);
           throw new Error('Completed, but results data missing from server response.');
        }
        break; // Exit polling loop
      } else if (statusData.status === 'failed' || statusData.status === 'error') {
        console.error(
          'Polling failed:',
          statusData.error || 'Unknown error'
        );
        throw new Error(`Categorization failed: ${statusData.error || 'Unknown error'}`);
      } else if (statusData.status === 'processing') {
        // Continue polling
      } else {
        console.warn(`Unexpected status encountered during polling: ${statusData.status}`);
        // Optional: throw an error or continue polling?
      }
    } catch (pollError) {
      console.error(`Error during polling attempt ${i + 1}:`, pollError.message);
      // Decide if we should retry or fail
      if (i === maxRetries - 1) {
        throw new Error(`Polling failed after ${maxRetries} attempts.`);
      }
    }
  }
  // Check if polling finished without getting results (e.g. timeout)
  if (!results) {
      throw new Error("Polling finished, but results were not obtained.")
  }
}

// --- Process final results ---
if (results) {
  console.log('\n--- Categorization Results ---');
  // ... existing code ...
} 