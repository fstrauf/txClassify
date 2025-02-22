"use client";

import { useUser } from "@auth0/nextjs-auth0/client";
import { LoginButton } from "../components/buttons/login-button";
import { LogoutButton } from "../components/buttons/logout-button";
import { SignupButton } from "../components/buttons/signup-button";
import Link from "next/link";

export const NavBarButtons = () => {
  const { user } = useUser();

  return (
    <div className="flex flex-col sm:flex-row items-center gap-3">
      {!user && (
        <>
          <SignupButton />
          <LoginButton />
        </>
      )}
      {user && (
        <div className="flex items-center gap-3">
          <LogoutButton />
          <Link 
            href="/profile" 
            className="flex items-center justify-center w-10 h-10 rounded-full overflow-hidden shadow-soft hover:shadow-glow transition-all duration-200"
          >
            <img
              src={user.picture}
              alt={user.name}
              className="w-full h-full object-cover"
            />
          </Link>
        </div>
      )}
    </div>
  );
};
