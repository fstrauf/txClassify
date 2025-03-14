"use client";
import { useState, useEffect } from "react";
import { usePostHog } from "posthog-js/react";
import { useUser } from "@auth0/nextjs-auth0/client";

export default function BetaAccessBanner() {
  const [email, setEmail] = useState("");
  const [showEmailInput, setShowEmailInput] = useState(false);
  const [submitted, setSubmitted] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState("");
  const [isBannerVisible, setIsBannerVisible] = useState(false);
  const [hasCheckedPreference, setHasCheckedPreference] = useState(false);
  const posthog = usePostHog();
  const { user, isLoading: isUserLoading } = useUser();

  // Check if the user has already opted in or dismissed the banner
  useEffect(() => {
    if (isUserLoading) {
      return;
    }

    if (!user) {
      setHasCheckedPreference(true);
      return;
    }

    const checkUserPreference = async () => {
      try {
        const response = await fetch("/api/user/getBetaPreference");
        const data = await response.json();

        if (response.ok && data.preference === null) {
          setTimeout(() => {
            setIsBannerVisible(true);
          }, 1000);
        }
      } catch (error) {
        console.error("Error checking beta preference:", error);
      } finally {
        setHasCheckedPreference(true);
      }
    };

    checkUserPreference();
  }, [user, isUserLoading]);

  // Track when the banner is viewed
  useEffect(() => {
    if (hasCheckedPreference && isBannerVisible && posthog) {
      posthog.capture("beta_app_banner_viewed");
    }
  }, [hasCheckedPreference, isBannerVisible, posthog]);

  const handleEmailSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError("");

    try {
      // Track the email submission event in PostHog
      posthog?.capture("beta_app_interest_email_submitted", {
        $set: { email: email }, // This will associate the email with the user
      });

      const response = await fetch("/api/createEmailContact", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          email,
          lists: ["441356"], // Using the landing page list ID
          tags: ["app_beta_interest"], // Add a tag to segment these subscribers
        }),
      });

      const data = await response.json();

      if (response.ok) {
        setSubmitted(true);
        // Track successful submission
        posthog?.capture("beta_app_interest_email_success");

        // Update user preference to OPTED_IN if logged in
        if (user) {
          await updateUserPreference("OPTED_IN");
        }
      } else {
        setError(data.error || "Failed to subscribe. Please try again.");
        // Track failed submission
        posthog?.capture("beta_app_interest_email_error", { error: data.error });
      }
    } catch (error) {
      setError("Something went wrong. Please try again.");
      // Track error
      posthog?.capture("beta_app_interest_email_error", { error: "Network error" });
    } finally {
      setIsLoading(false);
    }
  };

  const handleInterestClick = () => {
    // Track the initial interest click
    posthog?.capture("beta_app_interest_click");
    setShowEmailInput(true);
  };

  const handleDismiss = async () => {
    // Track when users dismiss the banner
    posthog?.capture("beta_app_interest_dismissed");
    setIsBannerVisible(false);

    // Update user preference to DISMISSED if logged in
    if (user) {
      await updateUserPreference("DISMISSED");
    }
  };

  // Helper function to update user preference
  const updateUserPreference = async (preference: "OPTED_IN" | "DISMISSED") => {
    try {
      await fetch("/api/user/updateBetaPreference", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ preference }),
      });
    } catch (error) {
      console.error("Error updating beta preference:", error);
    }
  };

  if (!hasCheckedPreference || !isBannerVisible) {
    return null;
  }

  return (
    <div className="fixed bottom-6 right-6 z-50 max-w-md animate-fadeIn">
      {/* Enhanced outer container with animated border and more pronounced shadow */}
      <div className="bg-gradient-to-r from-secondary via-secondary-light to-secondary p-[3px] rounded-lg shadow-xl animate-pulse">
        <div className="bg-gradient-to-br from-green-50 to-blue-50 dark:from-gray-800 dark:to-gray-900 rounded-lg overflow-hidden">
          <div className="relative p-5">
            {/* Banner header with icon */}
            <div className="absolute top-0 left-0 w-full bg-gradient-to-r from-secondary to-secondary-light py-1.5 px-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <svg
                    className="w-4 h-4 text-white"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      fillRule="evenodd"
                      d="M10 18a8 8 0 100-16 8 8 0 000 16zm1-11a1 1 0 10-2 0v2H7a1 1 0 100 2h2v2a1 1 0 102 0v-2h2a1 1 0 100-2h-2V7z"
                      clipRule="evenodd"
                    />
                  </svg>
                  <span className="text-xs font-bold text-white uppercase tracking-wider">Beta Access</span>
                </div>
                <button onClick={handleDismiss} className="text-white hover:text-gray-200" aria-label="Close">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
            </div>

            {/* Banner content - with extra padding for the header */}
            <div className="flex flex-col space-y-3 mt-6">
              <div className="flex items-center space-x-2">
                <span className="inline-flex items-center px-2.5 py-0.5 text-white rounded-full text-xs font-medium bg-secondary bg-opacity-20 text-secondary-dark dark:bg-secondary-dark dark:bg-opacity-30 dark:text-secondary-light">
                  Coming Soon
                </span>
                <h3 className="text-sm font-medium text-gray-900 dark:text-white">Web App Version</h3>
              </div>

              {!showEmailInput && !submitted && (
                <>
                  <p className="text-sm text-gray-700 dark:text-gray-300">
                    We're building a web version. Get beta access!
                  </p>
                  <button
                    onClick={handleInterestClick}
                    className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-secondary rounded-md hover:bg-secondary-dark focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-secondary transition-all duration-200 transform hover:scale-105"
                  >
                    <svg
                      className="w-4 h-4 mr-2"
                      fill="currentColor"
                      viewBox="0 0 20 20"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <path d="M10 12a2 2 0 100-4 2 2 0 000 4z" />
                      <path
                        fillRule="evenodd"
                        d="M.458 10C1.732 5.943 5.522 3 10 3s8.268 2.943 9.542 7c-1.274 4.057-5.064 7-9.542 7S1.732 14.057.458 10zM14 10a4 4 0 11-8 0 4 4 0 018 0z"
                        clipRule="evenodd"
                      />
                    </svg>
                    I'm interested
                  </button>
                </>
              )}

              {showEmailInput && !submitted && (
                <form onSubmit={handleEmailSubmit} className="space-y-3">
                  <div className="flex flex-col space-y-2">
                    <input
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      placeholder="Your email"
                      required
                      disabled={isLoading}
                      className="px-3 py-2 text-sm border border-gray-300 rounded-md focus:outline-none focus:ring-secondary focus:border-secondary dark:bg-gray-700 dark:border-gray-600 dark:text-white"
                    />
                    {error && <p className="text-xs text-red-500">{error}</p>}
                    <button
                      type="submit"
                      disabled={isLoading}
                      className="inline-flex items-center justify-center px-4 py-2 text-sm font-medium text-white bg-secondary rounded-md hover:bg-secondary-dark focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-secondary disabled:opacity-50 transition-all duration-200"
                    >
                      {isLoading ? (
                        <>
                          <svg
                            className="animate-spin -ml-1 mr-2 h-4 w-4 text-white"
                            xmlns="http://www.w3.org/2000/svg"
                            fill="none"
                            viewBox="0 0 24 24"
                          >
                            <circle
                              className="opacity-25"
                              cx="12"
                              cy="12"
                              r="10"
                              stroke="currentColor"
                              strokeWidth="4"
                            ></circle>
                            <path
                              className="opacity-75"
                              fill="currentColor"
                              d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"
                            ></path>
                          </svg>
                          Processing...
                        </>
                      ) : (
                        <>
                          <svg
                            className="w-4 h-4 mr-2"
                            fill="currentColor"
                            viewBox="0 0 20 20"
                            xmlns="http://www.w3.org/2000/svg"
                          >
                            <path d="M2.003 5.884L10 9.882l7.997-3.998A2 2 0 0016 4H4a2 2 0 00-1.997 1.884z" />
                            <path d="M18 8.118l-8 4-8-4V14a2 2 0 002 2h12a2 2 0 002-2V8.118z" />
                          </svg>
                          Get Early Access
                        </>
                      )}
                    </button>
                  </div>
                </form>
              )}

              {submitted && (
                <div className="text-center py-2">
                  <svg
                    className="w-10 h-10 text-secondary mx-auto mb-2"
                    fill="currentColor"
                    viewBox="0 0 20 20"
                    xmlns="http://www.w3.org/2000/svg"
                  >
                    <path
                      fillRule="evenodd"
                      d="M10 18a8 8 0 100-16 8 8 0 000 16zm3.707-9.293a1 1 0 00-1.414-1.414L9 10.586 7.707 9.293a1 1 0 00-1.414 1.414l2 2a1 1 0 001.414 0l4-4z"
                      clipRule="evenodd"
                    />
                  </svg>
                  <p className="text-sm text-secondary-dark dark:text-secondary-light font-medium">
                    Thanks! We'll notify you when the app is ready.
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
