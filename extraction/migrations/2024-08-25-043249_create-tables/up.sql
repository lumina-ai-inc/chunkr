-- Your SQL goes here
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

CREATE TABLE
  public.api_key_usage (
    id serial NOT NULL,
    api_key text NULL,
    usage integer NULL,
    usage_type text NULL,
    created_at timestamp with time zone NULL DEFAULT CURRENT_TIMESTAMP,
    service text NULL
  );

ALTER TABLE
  public.api_key_usage
ADD
  CONSTRAINT api_key_usage_pkey PRIMARY KEY (id);


CREATE TABLE
  public.api_key_limit (
    id serial NOT NULL,
    api_key text NULL,
    usage_limit integer NULL,
    usage_type text NULL,
    created_at timestamp with time zone NULL DEFAULT CURRENT_TIMESTAMP,
    service text NULL
  );

ALTER TABLE
  public.api_key_limit
ADD
  CONSTRAINT api_key_limit_pkey PRIMARY KEY (id);

CREATE TABLE
  public.api_users (
    key text NOT NULL,
    user_id text NULL,
    email text NULL,
    created_at timestamp with time zone NULL
  );

ALTER TABLE
  public.api_users
ADD
  CONSTRAINT api_users_pkey PRIMARY KEY (key);
