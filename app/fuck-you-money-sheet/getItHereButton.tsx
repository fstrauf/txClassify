"use client";
import { useState } from "react";

export default function GetItHereButton() {
  const [email, setEmail] = useState("");
  const [showEmailInput, setShowEmailInput] = useState(false);
  const [submitted, setSubmitted] = useState(false);

  const handleEmailSubmit = async (e: { preventDefault: () => void }) => {
    e.preventDefault();
    try {
      const body = { email };

      try {
        const response = await fetch("/api/createEmailContact", {
          method: "POST",
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });

        if (response.ok) {
          const data = await response.json();
          console.log("Fetched data:", data);
        } else {
          console.error("API call failed with status:", response.status);
        }
      } catch (error) {
        console.error("An error occurred:", error);
      }
      setSubmitted(true);
    } catch (error) {
      console.error("Error submitting email:", error);
    }
  };

  return (
    <div className="text-center">
      {/* {submitted ? ( */}
        <a
          className="bg-first hover:bg-second py-2 px-6 rounded-full text-white font-semibold transition duration-300 ease-in-out inline-block no-underline"
          href="https://docs.google.com/spreadsheets/d/1Buon6FEg7JGJMjuZgNgIrm5XyfP38JeaOJTNv6YQSHA/edit#gid=1128667954"
          target="_blank"
          rel="noopener noreferrer"
        >
          Download Spreadsheet
        </a>
      {/* ) : (
        <button
          onClick={() => setShowEmailInput(true)}
          className="bg-first hover:bg-second py-2 px-6 rounded-full text-white font-semibold transition duration-300 ease-in-out inline-block no-underline"
        >
          Get It Here
        </button>
      )} */}

      {/* {showEmailInput && !submitted && (
        <form onSubmit={handleEmailSubmit} className="mt-4">
          <p>We'd like to hear how you go!</p>
          <input
            type="email"
            placeholder="Enter your email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            className="p-2 border w-72 rounded mr-2 text-gray-600"
          />
          <button
            type="submit"
            className="bg-first hover:bg-second py-2 px-6 rounded-full text-white font-semibold transition duration-300 ease-in-out inline-block no-underline"
          >
            Submit
          </button>
        </form>
      )} */}
    </div>
  );
}
