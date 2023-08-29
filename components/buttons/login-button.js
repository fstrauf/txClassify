export const LoginButton = () => {
  const currentUrl = encodeURIComponent(window.location.pathname)

  return (
    <a
      className="bg-first hover:bg-second py-2 px-6 rounded-full text-white font-semibold transition duration-300 ease-in-out inline-block"
      href={`/api/auth/login?returnTo=${currentUrl}`}
      // onClick={handleLogin}
    >
      Log In
    </a>
  );
};
