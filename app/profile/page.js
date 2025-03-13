import { auth0 } from "@/src/lib/auth0";
import { redirect } from "next/navigation";
import ProfileClient from "./ProfileClient";

export default async function ProfilePage() {
  const session = await auth0.getSession();

  if (!session) {
    // Redirect to login if not authenticated
    redirect("/api/auth/login?returnTo=/profile");
  }

  return <ProfileClient user={session.user} />;
}
