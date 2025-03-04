import DatabaseExample from '../components/DatabaseExample';

export default function DbExamplePage() {
  return (
    <div className="container mx-auto py-8">
      <h1 className="text-2xl font-bold mb-6">Database Example with Drizzle ORM</h1>
      <p className="mb-6">
        This page demonstrates how to use Drizzle ORM with Supabase PostgreSQL for database operations.
      </p>
      <div className="bg-white rounded-lg shadow-md">
        <DatabaseExample />
      </div>
    </div>
  );
} 