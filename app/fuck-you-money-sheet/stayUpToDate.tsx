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
    <div className="bg-surface rounded-xl p-6 shadow-soft max-w-2xl mx-auto mt-8">
      <p className="text-gray-700 mb-3 text-center">Join our community of users:</p>
      <div className="flex justify-center">
        <a
          href="https://t.me/f_you_money"
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center text-primary hover:text-primary-dark font-semibold transition-colors"
        >
          <svg className="w-6 h-6 mr-2" fill="currentColor" viewBox="0 0 24 24">
            <path d="M12 0C5.373 0 0 5.373 0 12s5.373 12 12 12 12-5.373 12-12S18.627 0 12 0zm5.562 8.161c-.18 1.897-.962 6.502-1.359 8.627-.168.9-.5 1.201-.82 1.23-.697.064-1.226-.461-1.901-.903-1.056-.692-1.653-1.123-2.678-1.799-1.185-.781-.417-1.21.258-1.911.177-.184 3.247-2.977 3.307-3.23.007-.032.014-.15-.056-.212s-.174-.041-.249-.024c-.106.024-1.793 1.139-5.062 3.345-.479.329-.913.489-1.302.481-.428-.008-1.252-.241-1.865-.44-.752-.244-1.349-.374-1.297-.789.027-.216.325-.437.893-.663 3.498-1.524 5.831-2.529 6.998-3.014 3.332-1.386 4.025-1.627 4.476-1.635.099-.002.321.023.465.141.145.118.181.344.203.483.023.139.041.562.041.562z"/>
          </svg>
          @f_you_money on Telegram
        </a>
      </div>
    </div>
  );
}
