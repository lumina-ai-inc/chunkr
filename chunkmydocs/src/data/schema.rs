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

diesel::table! {
    ingestion_files (file_id) {
        id -> Nullable<Text>,
        file_id -> Text,
        task_id -> Nullable<Text>,
        file_name -> Nullable<Text>,
        file_size -> Nullable<Int8>,
        page_count -> Nullable<Int4>,
        created_at -> Nullable<Timestamptz>,
        status -> Nullable<Text>,
        input_location -> Nullable<Text>,
        output_location -> Nullable<Text>,
        expiration_time -> Nullable<Timestamptz>,
        model -> Nullable<Text>,
    }
}

diesel::table! {
    ingestion_tasks (task_id) {
        task_id -> Text,
        file_count -> Nullable<Int4>,
        total_size -> Nullable<Int8>,
        total_pages -> Nullable<Int4>,
        created_at -> Nullable<Timestamptz>,
        finished_at -> Nullable<Text>,
        api_key -> Nullable<Text>,
        status -> Nullable<Text>,
        url -> Nullable<Text>,
        model -> Nullable<Text>,
        expiration_time -> Nullable<Timestamptz>,
        message -> Nullable<Text>,
    }
}

diesel::table! {
    ingestion_usage (task_id) {
        task_id -> Text,
        user_id -> Nullable<Text>,
        api_key -> Nullable<Text>,
        usage_type -> Nullable<Text>,
        usage -> Nullable<Float8>,
        usage_unit -> Nullable<Text>,
        created_at -> Nullable<Timestamptz>,
    }
}

diesel::table! {
    usage_limit (id) {
        id -> Int4,
        user_id -> Nullable<Text>,
        usage_type -> Nullable<Text>,
        usage_limit -> Nullable<Float8>,
        usage_unit -> Nullable<Text>,
        created_at -> Nullable<Timestamptz>,
    }
}

diesel::joinable!(ingestion_files -> ingestion_tasks (task_id));
diesel::joinable!(ingestion_tasks -> api_keys (api_key));

diesel::allow_tables_to_appear_in_same_query!(
    api_key_limit,
    api_key_usage,
    api_keys,
    api_users,
    ingestion_files,
    ingestion_tasks,
    ingestion_usage,
    usage_limit,
);
