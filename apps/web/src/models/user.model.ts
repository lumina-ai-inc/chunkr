export interface Usage {
  usage: number;
  usage_limit: number;
  usage_type: "Segment" | "HighQuality" | "Fast";
  unit: "Segment" | "Page";
  created_at: string;
  updated_at: string;
}

export interface UserUsage {
  usage_type: "Fast" | "HighQuality" | "Segment";
  usage_limit: number;
  discounts: Discount[];
}

export interface Discount {
  usage_type: "Fast" | "HighQuality" | "Segment";
  amount: number;
}

export interface User {
  user_id: string;
  customer_id: string | null;
  email: string;
  first_name: string;
  last_name: string;
  task_count: number;
  api_keys: string[];
  tier: string;
  created_at: string;
  updated_at: string;
  usage: UserUsage[];
}

export interface AuthUser {
  access_token: string;
  profile: User;
}

export interface JWTAuthUser extends AuthUser {
  id_token: string;
  session_state: string;
  refresh_token: string;
  token_type: string;
  scope: string;
  expires_at: number;
  profile: JWTProfile;
}

export interface JWTProfile extends User {
  exp: number;
  iat: number;
  iss: string;
  aud: string;
  sub: string;
  typ: string;
  sid: string;
  email_verified: boolean;
  preferred_username: string;
  given_name: string;
  family_name: string;
}

export interface Discount {
  usage_type: "Fast" | "HighQuality" | "Segment";
  amount: number;
}
