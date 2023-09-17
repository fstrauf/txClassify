"use client";

import { useUser } from "@auth0/nextjs-auth0/client";
import { LoginButton } from "../components/buttons/login-button";
import { LogoutButton } from "../components/buttons/logout-button";
import { SignupButton } from "../components/buttons/signup-button";
import Link from "next/link";

export const NavBarButtons = () => {
  const { user } = useUser();

  return (
    <div className=" p-4 sm:p-1 flex flex-col justify-end sm:flex-row gap-2">
      {!user && (
        <>
          <SignupButton />
          <LoginButton />
        </>
      )}
      {user && (
        <div className="flex gap-2 ">
          <LogoutButton />
          <Link href="/profile" className="inline-block">
            <img
              src={user.picture}
              alt={user.name}
              className="rounded-full h-10 w-10 object-cover"
            />
          </Link>
        </div>
      )}
    </div>
  );
};
