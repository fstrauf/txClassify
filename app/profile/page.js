import { redirect } from "next/navigation";
import ProfileClient from "./ProfileClient";
import { getSession } from "@auth0/nextjs-auth0";

export default async function ProfilePage() {
  const session = await getSession();

  if (!session) {
    // Redirect to login if not authenticated
    redirect("/api/auth/login?returnTo=/profile");
  }

  return <ProfileClient user={session.user} />;
}
