"use client";

import { useUser } from "@auth0/nextjs-auth0/client";
import { LoginButton } from '../components/buttons/login-button'
import { LogoutButton } from "../components/buttons/logout-button";
import { SignupButton } from "../components/buttons/signup-button";

export const NavBarButtons = () => {
  const { user } = useUser();

  return (
    <div className="nav-bar__buttons">
      {!user && (
        <>
          <SignupButton />
          <LoginButton />
        </>
       )}
      {user && (
        <>
          <LogoutButton />
        </>
      )}
    </div>
  );
};
