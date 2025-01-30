import { useEffect } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useAuth } from "react-oidc-context";
import { getCheckoutSession } from "../../services/stripeService";
import { StripeCheckoutSession } from "../../models/stripe.models";
import toast from "react-hot-toast";
import Loader from "../Loader/Loader";

export default function Checkout() {
  const [searchParams] = useSearchParams();
  const sessionId = searchParams.get("session_id");
  const navigate = useNavigate();
  const auth = useAuth();

  useEffect(() => {
    const checkSession = async () => {
      try {
        if (!sessionId || !auth.user?.access_token) return;

        const session: StripeCheckoutSession = await getCheckoutSession(
          auth.user.access_token,
          sessionId
        );

        if (
          session.status === "complete" &&
          session.payment_status === "paid"
        ) {
          toast.success("Payment successful!");
          navigate("/dashboard");
        } else {
          toast.error("Payment unsuccessful");
          navigate("/pricing");
        }
      } catch (error) {
        console.error("Error:", error);
        toast.error("An error occurred while processing your payment");
        navigate("/pricing");
      }
    };

    checkSession();
  }, [sessionId, auth.user, navigate]);

  return (
    <div style={{ width: "100vw", height: "100vh" }}>
      <Loader />
    </div>
  );
}
