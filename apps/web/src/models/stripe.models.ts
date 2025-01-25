export interface StripeCheckoutSession {
  id: string;
  object: "checkout.session";
  amount_subtotal: number;
  amount_total: number;
  automatic_tax: {
    enabled: boolean;
    liability: null;
    status: null;
  };
  client_secret: string;
  created: number;
  currency: string;
  custom_fields: string[];
  custom_text: {
    after_submit: null;
    shipping_address: null;
    submit: null;
    terms_of_service_acceptance: null;
  };
  customer: string;
  customer_creation: null;
  customer_details: null;
  customer_email: null;
  expires_at: number;
  invoice: null;
  invoice_creation: null;
  livemode: boolean;
  locale: null;
  metadata: Record<string, string>;
  mode: "subscription" | "payment" | "setup";
  payment_intent: null;
  payment_link: null;
  payment_method_collection: "always" | "if_required";
  payment_method_configuration_details: {
    id: string;
    parent: null;
  };
  payment_method_options: {
    card: {
      request_three_d_secure: "automatic" | "any" | "never";
    };
  };
  payment_method_types: string[];
  payment_status: "paid" | "unpaid";
  phone_number_collection: {
    enabled: boolean;
  };
  shipping_options: string[];
  status: "open" | "complete" | "expired";
  success_url: string | null;
  total_details: {
    amount_discount: number;
    amount_shipping: number;
    amount_tax: number;
  };
  ui_mode: "embedded" | "hosted";
  url: string | null;
}
