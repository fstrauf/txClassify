import ExampleDatabaseComponent from '../components/ExampleDatabaseComponent';

export default function ExamplePage() {
  return (
    <div className="container mx-auto py-8">
      <h1 className="text-3xl font-bold mb-8 text-center">Drizzle ORM Example</h1>
      <ExampleDatabaseComponent />
    </div>
  );
} 