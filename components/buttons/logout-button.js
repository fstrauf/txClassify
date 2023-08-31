export const LogoutButton = () => {
  return (
    <a
      className="bg-first hover:bg-second py-1 px-3 rounded-full text-white font-semibold transition duration-300 ease-in-out flex items-center justify-center"
      href="/api/auth/logout"
    >
      Log Out
    </a>
  );
};
