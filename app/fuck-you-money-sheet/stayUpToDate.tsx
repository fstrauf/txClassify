"use client";
import { useState } from "react";

export default function StayUpToDate() {
  const [email, setEmail] = useState("");
  // const [showEmailInput, setShowEmailInput] = useState(false);
  const [statusMessage, setStatusMessage] = useState("");
  // const [submitted, setSubmitted] = useState(false);

  const handleEmailSubmit = async (e: { preventDefault: () => void }) => {
    e.preventDefault();
    try {
      const body = { email };

      try {
        const response = await fetch("/api/createEmailContact", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });

        if (response.ok) {
          const data = await response.json();
          console.log("Fetched data:", data);
          // setSubmitted(true);
          setStatusMessage("You're on the list");
        } else {
          console.error("API call failed with status:", response.status);
          setStatusMessage("Something went wrong, please try again later");
        }
      } catch (error) {
        console.error("An error occurred:", error);
        setStatusMessage("Something went wrong, please try again later");
      }
    } catch (error) {
      console.error("Error submitting email:", error);
      setStatusMessage("Something went wrong, please try again later");
    }
  };

  return (
    <div className="text-center">
      <form onSubmit={handleEmailSubmit} className="mt-4">
        <p>Stay up to date on changes to the template.</p>
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
      <p>{statusMessage}</p>
    </div>
  );
}
