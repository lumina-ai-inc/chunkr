-- Drop indexes on facts table
DROP INDEX IF EXISTS idx_facts_reviewed_by_user;
DROP INDEX IF EXISTS idx_facts_extraction_method;
DROP INDEX IF EXISTS idx_facts_confidence_score;

-- Remove columns from facts table
ALTER TABLE facts
DROP COLUMN IF EXISTS embedding_id,
DROP COLUMN IF EXISTS reviewed_by_user,
DROP COLUMN IF EXISTS extraction_method,
DROP COLUMN IF EXISTS confidence_score;

-- Drop index on documents table
DROP INDEX IF EXISTS idx_documents_fact_extraction_status;

-- Remove columns from documents table
ALTER TABLE documents
DROP COLUMN IF EXISTS embedding_id,
DROP COLUMN IF EXISTS fact_extraction_completed_at,
DROP COLUMN IF EXISTS fact_extraction_status,
DROP COLUMN IF EXISTS ocr_completed_at;

-- Drop investor_memos table and its indexes
DROP INDEX IF EXISTS idx_investor_memos_created_at;
DROP INDEX IF EXISTS idx_investor_memos_status;
DROP INDEX IF EXISTS idx_investor_memos_deal_id;
DROP TABLE IF EXISTS investor_memos;

-- Drop agent_executions table and its indexes
DROP INDEX IF EXISTS idx_agent_executions_created_at;
DROP INDEX IF EXISTS idx_agent_executions_agent_type;
DROP INDEX IF EXISTS idx_agent_executions_status;
DROP INDEX IF EXISTS idx_agent_executions_entity;
DROP TABLE IF EXISTS agent_executions;

-- Drop messages table and its indexes
DROP INDEX IF EXISTS idx_messages_role;
DROP INDEX IF EXISTS idx_messages_created_at;
DROP INDEX IF EXISTS idx_messages_conversation_id;
DROP TABLE IF EXISTS messages;

-- Drop conversations table and its indexes
DROP INDEX IF EXISTS idx_conversations_created_at;
DROP INDEX IF EXISTS idx_conversations_deal_id;
DROP INDEX IF EXISTS idx_conversations_user_id;
DROP TABLE IF EXISTS conversations;

