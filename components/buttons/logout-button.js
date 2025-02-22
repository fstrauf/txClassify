export const LogoutButton = () => {
  return (
    <a
      className="px-6 py-3 rounded-xl bg-white text-primary border border-primary/10 font-semibold hover:bg-gray-50 transition-all duration-200 shadow-soft flex items-center justify-center"
      href="/api/auth/logout"
    >
      Log Out
    </a>
  );
};
