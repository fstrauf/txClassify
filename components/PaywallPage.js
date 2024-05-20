import { useUser } from "@auth0/nextjs-auth0/client";
import UnauthorizedMessage from './Unauthorised';

const PaywallPage = ({ children }) => {
  const { user, error, isLoading } = useUser();
  console.log("🚀 ~ file: PaywallPage.js:6 ~ PaywallPage ~ user:", user)
  // find out what role the user hase: none, free, paid from supabase


  if (isLoading) {
    // Render a loading state here if needed
    return <div>Loading...</div>;
  }

  if (error) {
    // Handle error state if needed
    return <div>Error: {error.message}</div>;
  }

  return (
    <div>
      {/* <NavBarButtons /> */}
      {user ? children : <UnauthorizedMessage />}
    </div>
  );
};

export default PaywallPage;
