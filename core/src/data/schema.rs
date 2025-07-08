// @generated automatically by Diesel CLI.

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
        updated_at -> Nullable<Timestamptz>,
        expires_at -> Nullable<Timestamptz>,
        deleted_at -> Nullable<Timestamptz>,
        deleted_by -> Nullable<Text>,
    }
}

diesel::table! {
    invoices (invoice_id) {
        invoice_id -> Text,
        user_id -> Text,
        tasks -> Array<Nullable<Text>>,
        date_created -> Timestamp,
        date_paid -> Nullable<Timestamp>,
        invoice_status -> Text,
        amount_due -> Float8,
        total_pages -> Int4,
        stripe_invoice_id -> Nullable<Text>,
        bill_date -> Nullable<Timestamptz>,
    }
}

diesel::table! {
    monthly_usage (id) {
        id -> Int4,
        user_id -> Text,
        usage -> Nullable<Int4>,
        usage_type -> Text,
        year -> Int4,
        month -> Int4,
        created_at -> Nullable<Timestamptz>,
        updated_at -> Nullable<Timestamptz>,
        overage_usage -> Nullable<Int4>,
        tier -> Nullable<Text>,
        usage_limit -> Nullable<Int4>,
        billing_cycle_start -> Nullable<Timestamptz>,
        billing_cycle_end -> Nullable<Timestamptz>,
    }
}

diesel::table! {
    onboarding_records (id) {
        id -> Text,
        user_id -> Text,
        information -> Jsonb,
        status -> Text,
    }
}

diesel::table! {
    pre_applied_free_pages (id) {
        email -> Nullable<Text>,
        consumed -> Nullable<Bool>,
        usage_type -> Text,
        amount -> Int4,
        created_at -> Nullable<Timestamptz>,
        updated_at -> Nullable<Timestamptz>,
        id -> Int4,
    }
}

diesel::table! {
    segment_process (id) {
        id -> Text,
        user_id -> Nullable<Text>,
        task_id -> Nullable<Text>,
        segment_id -> Nullable<Text>,
        process_type -> Nullable<Text>,
        model_name -> Nullable<Text>,
        base_url -> Nullable<Text>,
        input_tokens -> Nullable<Int4>,
        output_tokens -> Nullable<Int4>,
        input_price -> Nullable<Float8>,
        output_price -> Nullable<Float8>,
        total_cost -> Nullable<Float8>,
        detail -> Nullable<Text>,
        latency -> Nullable<Float8>,
        avg_ocr_confidence -> Nullable<Float8>,
        created_at -> Nullable<Timestamp>,
    }
}

diesel::table! {
    subscriptions (user_id) {
        user_id -> Text,
        stripe_subscription_id -> Nullable<Text>,
        tier -> Text,
        last_paid_date -> Nullable<Timestamptz>,
        last_paid_status -> Nullable<Text>,
        created_at -> Nullable<Timestamptz>,
        updated_at -> Nullable<Timestamptz>,
    }
}

diesel::table! {
    task_invoices (task_id) {
        task_id -> Text,
        invoice_id -> Text,
        usage_type -> Text,
        pages -> Int4,
        cost -> Float8,
        created_at -> Timestamp,
        bill_date -> Nullable<Timestamptz>,
    }
}

diesel::table! {
    task_ledger (ledger_id) {
        ledger_id -> Text,
        task_id -> Text,
        user_id -> Text,
        tier -> Text,
        usage_type -> Text,
        tier_cost -> Float8,
        usage_amount -> Int4,
        total_cost -> Float8,
        created_at -> Timestamp,
    }
}

diesel::table! {
    tasks (task_id) {
        task_id -> Text,
        user_id -> Nullable<Text>,
        api_key -> Nullable<Text>,
        file_name -> Nullable<Text>,
        file_size -> Nullable<Int8>,
        page_count -> Nullable<Int4>,
        segment_count -> Nullable<Int4>,
        created_at -> Nullable<Timestamptz>,
        expires_at -> Nullable<Timestamptz>,
        finished_at -> Nullable<Timestamptz>,
        status -> Nullable<Text>,
        task_url -> Nullable<Text>,
        input_location -> Nullable<Text>,
        output_location -> Nullable<Text>,
        configuration -> Nullable<Text>,
        message -> Nullable<Text>,
        image_folder_location -> Nullable<Text>,
        pdf_location -> Nullable<Text>,
        #[max_length = 255]
        mime_type -> Nullable<Varchar>,
        started_at -> Nullable<Timestamptz>,
        #[max_length = 255]
        version -> Nullable<Varchar>,
    }
}

diesel::table! {
    tiers (tier) {
        tier -> Text,
        price_per_month -> Float8,
        usage_limit -> Int4,
        overage_rate -> Float8,
        created_at -> Nullable<Timestamptz>,
        updated_at -> Nullable<Timestamptz>,
    }
}

diesel::table! {
    usage (id) {
        id -> Int4,
        user_id -> Nullable<Text>,
        usage -> Nullable<Int4>,
        usage_type -> Nullable<Text>,
        unit -> Nullable<Text>,
        created_at -> Nullable<Timestamptz>,
        updated_at -> Nullable<Timestamptz>,
    }
}

diesel::table! {
    usage_type (id) {
        id -> Text,
        #[sql_name = "type"]
        type_ -> Text,
        description -> Text,
        unit -> Nullable<Text>,
        cost_per_unit_dollars -> Nullable<Float8>,
    }
}

diesel::table! {
    users (user_id) {
        user_id -> Text,
        customer_id -> Nullable<Text>,
        email -> Nullable<Text>,
        first_name -> Nullable<Text>,
        last_name -> Nullable<Text>,
        created_at -> Nullable<Timestamptz>,
        updated_at -> Nullable<Timestamptz>,
        tier -> Nullable<Text>,
        invoice_status -> Nullable<Text>,
        task_count -> Nullable<Int4>,
    }
}

diesel::joinable!(monthly_usage -> tiers (tier));
diesel::joinable!(onboarding_records -> users (user_id));
diesel::joinable!(subscriptions -> tiers (tier));
diesel::joinable!(subscriptions -> users (user_id));

diesel::allow_tables_to_appear_in_same_query!(
    api_keys,
    invoices,
    monthly_usage,
    onboarding_records,
    pre_applied_free_pages,
    segment_process,
    subscriptions,
    task_invoices,
    task_ledger,
    tasks,
    tiers,
    usage,
    usage_type,
    users,
);
