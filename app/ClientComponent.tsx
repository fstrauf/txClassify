"use client";

import { useAuth0User } from "@/components/Auth0ClientProvider";

export default function ClientComponent() {
  const user = useAuth0User();

  return (
    <div className="border p-4 rounded-lg bg-gray-50">
      <h2 className="text-xl font-semibold mb-2">Client Component</h2>
      {user ? (
        <div>
          <p>User from client context: {user.name}</p>
          {user.picture && (
            <div className="mt-2">
              <img src={user.picture} alt={`${user.name}'s profile`} className="w-12 h-12 rounded-full" />
            </div>
          )}
        </div>
      ) : (
        <p>No user in context</p>
      )}
    </div>
  );
}
