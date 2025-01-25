import { useEffect } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useAuth } from "react-oidc-context";
import { getCheckoutSession } from "../../services/stripeService";
import toast from "react-hot-toast";
import Loader from "../Loader/Loader";

export default function Checkout() {
  const [searchParams] = useSearchParams();
  const sessionId = searchParams.get("session_id");
  const navigate = useNavigate();
  const auth = useAuth();

  useEffect(() => {
    const pollSessionStatus = async () => {
      try {
        if (!sessionId || !auth.user?.access_token) {
          throw new Error("Missing session ID or access token");
        }

        const session = await getCheckoutSession(
          auth.user.access_token,
          sessionId
        );

        if (session.status === "complete") {
          toast.success(
            <div
              onClick={() =>
                navigate("/dashboard", { state: { selectedNav: "Usage" } })
              }
            >
              Subscription was successful! View new usage details
            </div>,
            {
              duration: 5000,
              style: { cursor: "pointer" },
            }
          );
          navigate("/dashboard", { state: { selectedNav: "Usage" } });
        } else if (session.status === "open") {
          // Payment failed or was canceled
          toast.error("Payment was not completed. Please try again.");
          navigate("/dashboard");
        }
      } catch (error) {
        console.error("Error checking session status:", error);
        toast.error("Something went wrong. Please try again.");
        navigate("/dashboard");
      }
    };

    pollSessionStatus();
  }, [sessionId, auth.user?.access_token, navigate]);

  return <Loader />;
}
