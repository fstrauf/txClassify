"use client";

import { useUser } from "@auth0/nextjs-auth0/client";
import Image from "next/image";

const Profile = () => {
  const defaultPicture =
    "https://cdn.auth0.com/blog/hello-auth0/auth0-user.png";
  const { user } = useUser();
  console.log("🚀 ~ file: page.js:10 ~ Profile ~ user:", user)

  if (!user) {
    return null;
  }

  return (
    <div className="bg-gradient-to-br from-first via-second to-third min-h-screen flex items-center justify-center">
      <div className="w-full max-w-md bg-third p-6 rounded-xl shadow-lg text-white">
        <h1 className="text-2xl font-semibold mb-4">Profile Page</h1>
        <div className="flex items-center">
          <Image
            src={user.picture || defaultPicture}
            alt="Profile"
            className="w-20 h-20 rounded-full mr-4"
            width={80}
            height={80}
          />
          <div>
            <h2 className="text-xl font-semibold">{user.name || user.nickname}</h2>
            <span className="text-white">{user.email}</span>
          </div>
        </div>
        <div className="mt-8 border-t pt-4">
          {/* Additional profile details can be added here */}
        </div>
      </div>
    </div>
  );
};

export default Profile;
