'use client';

import { useState, useEffect } from 'react';
import { createClient } from '@supabase/supabase-js';
import { toast, Toaster } from 'react-hot-toast';

const supabase = createClient(
  process.env.NEXT_PUBLIC_SUPABASE_URL!,
  process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY!
);

interface ApiKeyManagerProps {
  userId: string;
}

export default function ApiKeyManager({ userId }: ApiKeyManagerProps) {
  const [apiKey, setApiKey] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchApiKey();
  }, []);

  const fetchApiKey = async () => {
    try {
      const { data, error } = await supabase
        .from('account')
        .select('api_key')
        .eq('userId', userId)
        .single();

      if (error) throw error;
      setApiKey(data?.api_key || null);
    } catch (error) {
      console.error('Error fetching API key:', error);
      toast.error('Failed to fetch API key');
    } finally {
      setLoading(false);
    }
  };

  const generateApiKey = async () => {
    try {
      const newApiKey = crypto.randomUUID();
      
      const { error } = await supabase
        .from('account')
        .upsert({ 
          userId: userId,
          api_key: newApiKey
        });

      if (error) throw error;
      
      setApiKey(newApiKey);
      toast.success('API key generated successfully');
    } catch (error) {
      console.error('Error generating API key:', error);
      toast.error('Failed to generate API key');
    }
  };

  const copyToClipboard = async () => {
    try {
      await navigator.clipboard.writeText(apiKey!);
      toast.success('API key copied to clipboard');
    } catch (error) {
      toast.error('Failed to copy API key');
    }
  };

  if (loading) {
    return <div className="text-center">Loading...</div>;
  }

  return (
    <div className="space-y-6">
      <Toaster />
      {apiKey ? (
        <div className="space-y-4">
          <div className="bg-gray-800 p-4 rounded-lg break-all">
            <p className="text-sm font-mono">{apiKey}</p>
          </div>
          <div className="flex gap-4 justify-center">
            <button
              onClick={copyToClipboard}
              className="bg-first hover:bg-second py-2 px-6 rounded-full text-white font-semibold transition duration-300 ease-in-out"
            >
              Copy API Key
            </button>
            <button
              onClick={generateApiKey}
              className="bg-first hover:bg-second py-2 px-6 rounded-full text-white font-semibold transition duration-300 ease-in-out"
            >
              Generate New Key
            </button>
          </div>
        </div>
      ) : (
        <div className="text-center">
          <button
            onClick={generateApiKey}
            className="bg-first hover:bg-second py-2 px-6 rounded-full text-white font-semibold transition duration-300 ease-in-out"
          >
            Generate API Key
          </button>
        </div>
      )}
      <div className="mt-8 text-sm text-gray-300">
        <h2 className="font-semibold mb-2">How to use your API key:</h2>
        <ol className="list-decimal list-inside space-y-2">
          <li>Copy your API key</li>
          <li>Open your Google Sheets™ document</li>
          <li>Go to Extensions &gt; Apps Script</li>
          <li>Add your API key to the configuration</li>
          <li>Save and reload your sheet</li>
        </ol>
      </div>
    </div>
  );
} 