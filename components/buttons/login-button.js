import { usePathname } from 'next/navigation';

export const LoginButton = () => {
  const pathname = usePathname()

  return (
    <a
      className="px-6 py-3 rounded-xl bg-white text-primary border border-primary/10 font-semibold hover:bg-gray-50 transition-all duration-200 shadow-soft"
      href={`/api/auth/login?returnTo=${pathname}`}
    >
      Log In
    </a>
  );
};