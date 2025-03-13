"use client";

import { createContext, useContext, ReactNode } from "react";

// Define the user type based on Auth0 user structure
type User = {
  sub?: string;
  name?: string;
  email?: string;
  picture?: string;
  [key: string]: any;
};

// Create context with default undefined value
const UserContext = createContext<User | undefined>(undefined);

// Provider component
export function Auth0ClientProvider({ user, children }: { user: User | undefined; children: ReactNode }) {
  return <UserContext.Provider value={user}>{children}</UserContext.Provider>;
}

// Hook to use the auth context
export function useAuth0User() {
  const context = useContext(UserContext);
  return context;
}
