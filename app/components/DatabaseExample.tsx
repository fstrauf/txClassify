'use client';

import { useState, useEffect } from 'react';

interface Account {
  userId: string;
  apiKey: string | null;
  categorisationRange: string | null;
  categorisationTab: string | null;
}

export default function DatabaseExample() {
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchAccounts();
  }, []);

  const fetchAccounts = async () => {
    try {
      setLoading(true);
      
      // Fetch accounts using API route
      const response = await fetch('/api/account');
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to fetch accounts');
      }
      
      const data = await response.json();
      setAccounts(data);
    } catch (err: any) {
      console.error('Error fetching accounts:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const createTestAccount = async () => {
    try {
      setLoading(true);
      
      // Generate a random user ID
      const randomId = `user_${Math.random().toString(36).substring(2, 9)}`;
      
      // Create a test account using API route
      const response = await fetch('/api/account', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          userId: randomId,
          apiKey: `api_${Math.random().toString(36).substring(2, 9)}`,
          categorisationRange: 'A:Z',
          categorisationTab: 'Sheet1',
          columnOrderCategorisation: { categoryColumn: 'E', descriptionColumn: 'C' }
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to create account');
      }
      
      // Refresh the accounts list
      fetchAccounts();
    } catch (err: any) {
      console.error('Error creating test account:', err);
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold">Accounts</h2>
        <button
          onClick={createTestAccount}
          className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
        >
          Create Test Account
        </button>
      </div>
      
      {error && (
        <div className="p-4 bg-red-100 text-red-700 rounded">
          Error: {error}
        </div>
      )}
      
      {loading ? (
        <div className="p-4 text-center">Loading...</div>
      ) : accounts.length === 0 ? (
        <div className="p-4 text-center">No accounts found</div>
      ) : (
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {accounts.map((account) => (
            <div key={account.userId} className="p-4 border rounded shadow">
              <h3 className="font-bold">{account.userId}</h3>
              <p>API Key: {account.apiKey || 'None'}</p>
              <p>Range: {account.categorisationRange || 'Default'}</p>
              <p>Tab: {account.categorisationTab || 'Default'}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  );
} 