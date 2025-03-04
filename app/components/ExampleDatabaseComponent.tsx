'use client';

import { useState, useEffect } from 'react';

// Define types based on our schema
type Account = {
  id?: number;
  userId: string;
  categorisationRange?: string;
  categorisationTab?: string;
  columnOrderCategorisation?: {
    categoryColumn: string;
    descriptionColumn: string;
  };
  apiKey?: string;
};

export default function ExampleDatabaseComponent() {
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [newAccount, setNewAccount] = useState<Account>({
    userId: '',
    categorisationRange: 'A:Z',
    categorisationTab: 'Sheet1',
    columnOrderCategorisation: {
      categoryColumn: 'E',
      descriptionColumn: 'C',
    },
    apiKey: '',
  });

  // Fetch accounts on component mount
  useEffect(() => {
    fetchAccounts();
  }, []);

  // Function to fetch accounts from API
  const fetchAccounts = async () => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/example');
      const result = await response.json();
      
      if (result.success) {
        setAccounts(result.data);
      } else {
        setError(result.error || 'Failed to fetch accounts');
      }
    } catch (err) {
      setError('Error fetching accounts: ' + (err instanceof Error ? err.message : String(err)));
    } finally {
      setLoading(false);
    }
  };

  // Function to create a new account
  const createAccount = async (e: React.FormEvent) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch('/api/example', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(newAccount),
      });
      
      const result = await response.json();
      
      if (result.success) {
        // Reset form and refresh accounts
        setNewAccount({
          userId: '',
          categorisationRange: 'A:Z',
          categorisationTab: 'Sheet1',
          columnOrderCategorisation: {
            categoryColumn: 'E',
            descriptionColumn: 'C',
          },
          apiKey: '',
        });
        fetchAccounts();
      } else {
        setError(result.error || 'Failed to create account');
      }
    } catch (err) {
      setError('Error creating account: ' + (err instanceof Error ? err.message : String(err)));
    } finally {
      setLoading(false);
    }
  };

  // Function to delete an account
  const deleteAccount = async (userId: string) => {
    setLoading(true);
    setError(null);
    
    try {
      const response = await fetch(`/api/example?userId=${encodeURIComponent(userId)}`, {
        method: 'DELETE',
      });
      
      const result = await response.json();
      
      if (result.success) {
        fetchAccounts();
      } else {
        setError(result.error || 'Failed to delete account');
      }
    } catch (err) {
      setError('Error deleting account: ' + (err instanceof Error ? err.message : String(err)));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <h1 className="text-2xl font-bold mb-6">Database Example</h1>
      
      {/* Error display */}
      {error && (
        <div className="bg-red-100 border border-red-400 text-red-700 px-4 py-3 rounded mb-4">
          <p>{error}</p>
        </div>
      )}
      
      {/* Create account form */}
      <div className="bg-white shadow-md rounded px-8 pt-6 pb-8 mb-6">
        <h2 className="text-xl font-semibold mb-4">Create New Account</h2>
        <form onSubmit={createAccount}>
          <div className="mb-4">
            <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="userId">
              User ID
            </label>
            <input
              className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
              id="userId"
              type="text"
              value={newAccount.userId}
              onChange={(e) => setNewAccount({ ...newAccount, userId: e.target.value })}
              required
            />
          </div>
          <div className="mb-4">
            <label className="block text-gray-700 text-sm font-bold mb-2" htmlFor="apiKey">
              API Key
            </label>
            <input
              className="shadow appearance-none border rounded w-full py-2 px-3 text-gray-700 leading-tight focus:outline-none focus:shadow-outline"
              id="apiKey"
              type="text"
              value={newAccount.apiKey}
              onChange={(e) => setNewAccount({ ...newAccount, apiKey: e.target.value })}
            />
          </div>
          <div className="flex items-center justify-between">
            <button
              className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
              type="submit"
              disabled={loading}
            >
              {loading ? 'Creating...' : 'Create Account'}
            </button>
            <button
              className="bg-gray-500 hover:bg-gray-700 text-white font-bold py-2 px-4 rounded focus:outline-none focus:shadow-outline"
              type="button"
              onClick={fetchAccounts}
              disabled={loading}
            >
              Refresh Accounts
            </button>
          </div>
        </form>
      </div>
      
      {/* Accounts list */}
      <div className="bg-white shadow-md rounded px-8 pt-6 pb-8">
        <h2 className="text-xl font-semibold mb-4">Accounts</h2>
        {loading ? (
          <p>Loading...</p>
        ) : accounts.length === 0 ? (
          <p>No accounts found.</p>
        ) : (
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white">
              <thead>
                <tr>
                  <th className="py-2 px-4 border-b border-gray-200 bg-gray-50 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                    User ID
                  </th>
                  <th className="py-2 px-4 border-b border-gray-200 bg-gray-50 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                    API Key
                  </th>
                  <th className="py-2 px-4 border-b border-gray-200 bg-gray-50 text-left text-xs font-semibold text-gray-600 uppercase tracking-wider">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody>
                {accounts.map((account) => (
                  <tr key={account.userId}>
                    <td className="py-2 px-4 border-b border-gray-200">{account.userId}</td>
                    <td className="py-2 px-4 border-b border-gray-200">{account.apiKey}</td>
                    <td className="py-2 px-4 border-b border-gray-200">
                      <button
                        className="bg-red-500 hover:bg-red-700 text-white font-bold py-1 px-2 rounded text-xs"
                        onClick={() => deleteAccount(account.userId)}
                        disabled={loading}
                      >
                        Delete
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
} 