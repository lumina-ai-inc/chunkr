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
    deals (deal_id) {
        deal_id -> Text,
        user_id -> Text,
        deal_name -> Text,
        status -> Text,
        created_at -> Timestamptz,
        updated_at -> Timestamptz,
        metadata -> Jsonb,
    }
}

diesel::table! {
    documents (document_id) {
        document_id -> Text,
        deal_id -> Text,
        file_name -> Text,
        document_type -> Text,
        status -> Text,
        storage_location -> Nullable<Text>,
        page_count -> Nullable<Int4>,
        ocr_output -> Nullable<Jsonb>,
        created_at -> Timestamptz,
        ocr_completed_at -> Nullable<Timestamptz>,
        fact_extraction_status -> Nullable<Text>,
        fact_extraction_completed_at -> Nullable<Timestamptz>,
        embedding_id -> Nullable<Text>,
    }
}

diesel::table! {
    facts (fact_id) {
        fact_id -> Text,
        document_id -> Text,
        deal_id -> Text,
        fact_type -> Text,
        label -> Text,
        value -> Text,
        unit -> Nullable<Text>,
        source_citation -> Jsonb,
        status -> Text,
        confidence_score -> Nullable<Float8>,
        approved_at -> Nullable<Timestamptz>,
        approved_by -> Nullable<Text>,
        locked -> Bool,
        created_at -> Timestamptz,
        extraction_method -> Nullable<Text>,
        reviewed_by_user -> Nullable<Bool>,
        embedding_id -> Nullable<Text>,
    }
}

diesel::table! {
    discounts (user_id, usage_type) {
        user_id -> Text,
        usage_type -> Text,
        amount -> Nullable<Float8>,
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
    task_invoices (task_id) {
        task_id -> Text,
        invoice_id -> Text,
        usage_type -> Text,
        pages -> Int4,
        cost -> Float8,
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
        pdf_location -> Nullable<Text>,
        input_file_type -> Nullable<Text>,
        #[max_length = 255]
        mime_type -> Nullable<Varchar>,
        started_at -> Nullable<Timestamptz>,
        #[max_length = 255]
        image_folder_location -> Nullable<Varchar>,
    }
}

diesel::table! {
    usage (id) {
        id -> Int4,
        user_id -> Nullable<Text>,
        #[sql_name = "usage"]
        usage_col -> Nullable<Int4>,
        usage_limit -> Nullable<Int4>,
        usage_type -> Nullable<Text>,
        unit -> Nullable<Text>,
        created_at -> Nullable<Timestamptz>,
        updated_at -> Nullable<Timestamptz>,
    }
}

diesel::table! {
    usage_limits (id) {
        id -> Int4,
        usage_type -> Text,
        tier -> Text,
        usage_limit -> Int4,
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

diesel::table! {
    conversations (conversation_id) {
        conversation_id -> Text,
        user_id -> Text,
        deal_id -> Nullable<Text>,
        title -> Nullable<Text>,
        context -> Nullable<Jsonb>,
        created_at -> Timestamptz,
        updated_at -> Timestamptz,
    }
}

diesel::table! {
    messages (message_id) {
        message_id -> Text,
        conversation_id -> Text,
        role -> Text,
        content -> Text,
        metadata -> Nullable<Jsonb>,
        embedding_id -> Nullable<Text>,
        created_at -> Timestamptz,
    }
}

diesel::table! {
    agent_executions (execution_id) {
        execution_id -> Text,
        agent_type -> Text,
        entity_id -> Nullable<Text>,
        entity_type -> Nullable<Text>,
        input -> Jsonb,
        output -> Nullable<Jsonb>,
        status -> Text,
        error -> Nullable<Text>,
        llm_provider -> Nullable<Text>,
        model -> Nullable<Text>,
        tokens_used -> Nullable<Int4>,
        execution_time_ms -> Nullable<Int4>,
        created_at -> Timestamptz,
        completed_at -> Nullable<Timestamptz>,
    }
}

diesel::table! {
    investor_memos (memo_id) {
        memo_id -> Text,
        deal_id -> Text,
        title -> Nullable<Text>,
        content -> Text,
        sections -> Nullable<Jsonb>,
        version -> Nullable<Int4>,
        status -> Nullable<Text>,
        created_by_agent -> Nullable<Bool>,
        created_at -> Timestamptz,
        updated_at -> Timestamptz,
    }
}

diesel::joinable!(conversations -> deals (deal_id));
diesel::joinable!(conversations -> users (user_id));
diesel::joinable!(deals -> users (user_id));
diesel::joinable!(documents -> deals (deal_id));
diesel::joinable!(facts -> deals (deal_id));
diesel::joinable!(facts -> documents (document_id));
diesel::joinable!(investor_memos -> deals (deal_id));
diesel::joinable!(messages -> conversations (conversation_id));

diesel::allow_tables_to_appear_in_same_query!(
    agent_executions,
    api_keys,
    conversations,
    deals,
    discounts,
    documents,
    facts,
    investor_memos,
    invoices,
    messages,
    monthly_usage,
    pre_applied_free_pages,
    segment_process,
    task_invoices,
    tasks,
    usage,
    usage_limits,
    usage_type,
    users,
);
