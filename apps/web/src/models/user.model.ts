export interface Usage {
  usage: number;
  usage_limit: number;
  usage_type: "Segment" | "HighQuality" | "Fast";
  unit: "Segment" | "Page";
  created_at: string;
  updated_at: string;
}

export interface User {
  user_id: string;
  customer_id: string | null;
  email: string;
  first_name: string;
  last_name: string;
  api_keys: string[];
  tier: string;
  created_at: string;
  updated_at: string;
  usages: Usage[];
}
