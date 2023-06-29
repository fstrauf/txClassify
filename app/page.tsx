import UserInput from "./UserInput";

export default function Home() {

  return (
    <main className="flex min-h-screen flex-col items-center justify-between p-24">
      <div className="w-full items-center justify-between font-mono text-sm lg:flex">
        <UserInput/>
      </div>
    </main>
  );
}
