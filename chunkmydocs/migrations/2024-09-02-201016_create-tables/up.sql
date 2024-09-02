-- Your SQL goes here
CREATE TABLE
  public.users (
    user_id text NULL,
    email text NULL,
    created_at timestamp with time zone NULL,
    usage_tier text NULL,
    usage_limit integer NULL,
    service text NULL,
    usage integer NULL
  );

ALTER TABLE
  public.users
ADD
  CONSTRAINT users_pkey PRIMARY KEY (user_id);


CREATE TABLE
  public.api_keys (
    key text NOT NULL,
    user_id text NULL,
    dataset_id text NULL,
    org_id text NULL,
    access_level text NULL,
    active boolean NULL,
    deleted boolean NULL,
    created_at timestamp with time zone NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at timestamp with time zone NULL,
    deleted_at timestamp with time zone NULL,
    deleted_by text NULL
  );

ALTER TABLE
  public.api_keys
ADD
  CONSTRAINT api_keys_pkey PRIMARY KEY (key);