export interface UserUsage {
  usage_type: "Page";
  usage_limit: number;
  overage_usage?: number;
}

export enum OnboardingStatus {
  Pending = "Pending",
  Completed = "Completed",
}

interface OnboardingInformation {
  use_case: string;
  usage: string;
  file_types: string;
}

interface OnboardingRecord {
  id: string;
  status: OnboardingStatus;
  information: OnboardingInformation;
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
  last_paid_status: string | null;
  onboarding_record: OnboardingRecord | null;
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
