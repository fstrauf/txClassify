import React, { useState } from "react";
// import Instructions from "./instructions";
// import Intro from "./intro";

export default function EmailPopover({ onClose }) {
  const [email, setEmail] = useState("");

  const handleSubmit = (e: { preventDefault: () => void; }) => {
    e.preventDefault();
    // Perform email submission logic here
    // Once the email is submitted, you can call onClose to close the popover
    onClose();
  };

  return (
    <div className="bg-white border p-4 rounded shadow-md">
      <div className="font-bold mb-2">Email Subscription</div>
      <div>
        Please provide your email to receive the download link:
        <form onSubmit={handleSubmit} className="mt-2">
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            className="w-full p-2 border rounded"
          />
          <button
            type="submit"
            className="mt-2 px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            Submit
          </button>
        </form>
      </div>
    </div>
  );
}
