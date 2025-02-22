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
    return <div className="text-gray-700 text-center">Loading...</div>;
  }

  return (
    <div className="space-y-8">
      <Toaster />
      {apiKey ? (
        <div className="space-y-6">
          <div className="bg-gray-50 p-6 rounded-xl shadow-soft border border-gray-200">
            <p className="text-sm font-mono text-gray-700 break-all">{apiKey}</p>
          </div>
          <div className="flex flex-wrap gap-4 justify-center">
            <button
              onClick={copyToClipboard}
              className="px-6 py-3 rounded-xl bg-primary text-white font-semibold hover:bg-primary-dark transition-all duration-200 shadow-soft hover:shadow-glow"
            >
              Copy API Key
            </button>
            <button
              onClick={generateApiKey}
              className="px-6 py-3 rounded-xl bg-white text-primary border border-primary/10 font-semibold hover:bg-gray-50 transition-all duration-200 shadow-soft"
            >
              Generate New Key
            </button>
          </div>
        </div>
      ) : (
        <div className="text-center">
          <button
            onClick={generateApiKey}
            className="px-6 py-3 rounded-xl bg-primary text-white font-semibold hover:bg-primary-dark transition-all duration-200 shadow-soft hover:shadow-glow"
          >
            Generate API Key
          </button>
        </div>
      )}
      <div className="bg-surface rounded-xl p-6 shadow-soft">
        <h2 className="text-xl font-semibold text-gray-900 mb-4">How to use your API key:</h2>
        <ol className="list-decimal list-inside space-y-3 text-gray-700">
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