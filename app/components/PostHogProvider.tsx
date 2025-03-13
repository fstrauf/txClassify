"use client";

import posthog from "posthog-js";
import { PostHogProvider } from "posthog-js/react";
import { useEffect } from "react";
import { useUser } from "@auth0/nextjs-auth0/client";
import PostHogPageView from "./PostHogPageView";

export default function PostHogProviderWrapper({ children }: { children: React.ReactNode }) {
  const { user, isLoading } = useUser();

  useEffect(() => {
    posthog.init(process.env.NEXT_PUBLIC_POSTHOG_KEY || "", {
      api_host: process.env.NEXT_PUBLIC_POSTHOG_HOST || "https://us.i.posthog.com",
      capture_pageview: false, // Disable automatic pageview capture, as we capture manually
      person_profiles: "identified_only", // Only create profiles for identified users
      loaded: (posthog) => {
        if (process.env.NODE_ENV === "development") posthog.debug();
      },
    });
  }, []);

  // Identify user when they're logged in or reset when logged out
  useEffect(() => {
    if (!isLoading) {
      if (user && user.email && user.sub) {
        // Collect user properties for identification
        const userProperties = {
          email: user.email,
          name: user.name || "",
          nickname: user.nickname || "",
          picture: user.picture || "",
          // Add any other relevant user properties from Auth0
          auth0_updated_at: user.updated_at,
          email_verified: user.email_verified,
        };

        // Identify the user with their email and properties
        posthog.identify(user.sub, userProperties);

        // Log an event when a user is identified
        posthog.capture("user_identified");
      } else {
        // Reset the user identification when logged out
        posthog.reset();
      }
    }
  }, [user, isLoading]);

  return (
    <PostHogProvider client={posthog}>
      <PostHogPageView />
      {children}
    </PostHogProvider>
  );
}
