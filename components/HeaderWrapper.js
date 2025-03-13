import { auth0 } from "@/src/lib/auth0";
import Header from "./Header";

export default async function HeaderWrapper() {
  const session = await auth0.getSession();
  const user = session?.user;

  return <Header user={user} />;
}
