// @generated automatically by Diesel CLI.

diesel::table! {
    api_key_limit (id) {
        id -> Int4,
        api_key -> Nullable<Text>,
        usage_limit -> Nullable<Int4>,
        usage_type -> Nullable<Text>,
        created_at -> Nullable<Timestamptz>,
        service -> Nullable<Text>,
    }
}

diesel::table! {
    api_key_usage (id) {
        id -> Int4,
        api_key -> Nullable<Text>,
        usage -> Nullable<Int4>,
        usage_type -> Nullable<Text>,
        created_at -> Nullable<Timestamptz>,
        service -> Nullable<Text>,
    }
}

diesel::table! {
    api_keys (key) {
        key -> Text,
        user_id -> Nullable<Text>,
        dataset_id -> Nullable<Text>,
        org_id -> Nullable<Text>,
        access_level -> Nullable<Text>,
        active -> Nullable<Bool>,
        deleted -> Nullable<Bool>,
        created_at -> Nullable<Timestamptz>,
        expires_at -> Nullable<Timestamptz>,
        deleted_at -> Nullable<Timestamptz>,
        deleted_by -> Nullable<Text>,
    }
}

diesel::table! {
    api_users (key) {
        key -> Text,
        user_id -> Nullable<Text>,
        email -> Nullable<Text>,
        created_at -> Nullable<Timestamptz>,
    }
}

diesel::allow_tables_to_appear_in_same_query!(
    api_key_limit,
    api_key_usage,
    api_keys,
    api_users,
);
