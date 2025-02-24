'use client'

import posthog from 'posthog-js'
import { PostHogProvider } from 'posthog-js/react'
import { useEffect } from 'react'
import PostHogPageView from "./PostHogPageView"

export default function PostHogProviderWrapper({
  children,
}: {
  children: React.ReactNode
}) {
  useEffect(() => {
    posthog.init(process.env.NEXT_PUBLIC_POSTHOG_KEY || '', {
      api_host: process.env.NEXT_PUBLIC_POSTHOG_HOST || 'https://us.i.posthog.com',
      capture_pageview: false, // Disable automatic pageview capture, as we capture manually
      person_profiles: 'identified_only', // Only create profiles for identified users
      loaded: (posthog) => {
        if (process.env.NODE_ENV === 'development') posthog.debug()
      }
    })
  }, [])

  return (
    <PostHogProvider client={posthog}>
      <PostHogPageView />
      {children}
    </PostHogProvider>
  )
} 