import { createPortal } from "react-dom";
import {
  EmbeddedCheckoutProvider,
  EmbeddedCheckout,
} from "@stripe/react-stripe-js";
import { Stripe } from "@stripe/stripe-js";

interface CheckoutOverlayProps {
  onClose: () => void;
  stripePromise: Promise<Stripe | null>;
  clientSecret: string;
}

const CheckoutOverlay = ({
  onClose,
  stripePromise,
  clientSecret,
}: CheckoutOverlayProps) => {
  return createPortal(
    <div className="checkout-overlay" onClick={onClose}>
      <div className="checkout-popover" onClick={(e) => e.stopPropagation()}>
        <EmbeddedCheckoutProvider
          stripe={stripePromise}
          options={{ clientSecret }}
        >
          <div className="stripe-checkout-wrapper">
            <EmbeddedCheckout />
          </div>
        </EmbeddedCheckoutProvider>
      </div>
    </div>,
    document.body
  );
};

export default CheckoutOverlay;
