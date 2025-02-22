import { getSession } from "@auth0/nextjs-auth0";
import ApiKeyManager from "./ApiKeyManager";

export default async function ApiKeyPage() {
  const session = await getSession();
  const user = session?.user;

  return (
    <div className="flex flex-col min-h-screen bg-gradient-to-br from-first via-second to-third">
      <main className="flex-grow flex items-center justify-center p-10">
        <div className="w-full max-w-4xl bg-third p-6 rounded-xl shadow-lg text-white space-y-6">
          <h1 className="text-3xl font-bold leading-tight text-center">
            API Key Management
          </h1>
          <p className="text-lg text-center">
            Generate and manage your API key to use with the Google Sheets™ integration.
          </p>
          {user ? (
            <ApiKeyManager userId={user.sub} />
          ) : (
            <p className="text-center">Please log in to manage your API key.</p>
          )}
        </div>
      </main>
    </div>
  );
} 