"use client";
import { useState } from "react";

export default function UserInput() {
  const [selectedFile, setSelectedFile] = useState(null);

  const handleFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
  };

  const handleFormSubmit = (event) => {
    event.preventDefault();

    if (selectedFile) {
      const formData = new FormData();
      formData.append("file", selectedFile);

      // Send the form data to the API endpoint
      fetch("/api/classify", {
        method: "POST",
        body: formData,
      })
        .then((response) => {
            console.log("🚀 ~ file: UserInput.tsx:24 ~ handleFormSubmit ~ response:", response)
            return response.json()
        })
        .then((data) => {
          // Handle the API response
          console.log(data);
        })
        .catch((error) => {
          // Handle any error that occurred during the API call
          console.error(error);
        });
    }
  };

  return (
    <form onSubmit={handleFormSubmit}>
      <div className="flex flex-col">
        <label htmlFor="fileUpload" className="block mb-4">
          Upload new transactions to be classified:
        </label>
        <input
          type="file"
          id="fileUpload"
          accept=".csv"
          onChange={handleFileChange}
          className="mb-4"
        />
        <button
          type="submit"
          className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
        >
          Classify
        </button>
      </div>
    </form>
  );
}
