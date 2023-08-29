import { LoginButton } from '../components/buttons/login-button';
import { SignupButton } from '../components/buttons/signup-button';

const UnauthorizedMessage = () => {
  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-100">
      <h1 className="text-3xl font-bold mb-4">You Need to Login</h1>
      <p className="text-lg mb-8">Please login to access this page.</p>
      <div className="flex gap-4">
        <SignupButton />
        <LoginButton />
      </div>
    </div>
  );
};

export default UnauthorizedMessage;
