import { useUser } from "@auth0/nextjs-auth0/client";
import UnauthorizedMessage from './Unauthorised';

const ProtectedPage = ({ children }) => {
  const { user, error, isLoading } = useUser();

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

export default ProtectedPage;
