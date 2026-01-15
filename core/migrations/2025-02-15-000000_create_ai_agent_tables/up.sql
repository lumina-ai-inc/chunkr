-- Create conversations table for AI chat history
CREATE TABLE conversations (
    conversation_id VARCHAR(255) PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    deal_id VARCHAR(255), -- nullable, can exist without deal
    title VARCHAR(500),
    context JSONB, -- conversation metadata
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
    FOREIGN KEY (deal_id) REFERENCES deals(deal_id) ON DELETE SET NULL
);

CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_conversations_deal_id ON conversations(deal_id);
CREATE INDEX idx_conversations_created_at ON conversations(created_at DESC);

-- Create messages table for individual chat messages
CREATE TABLE messages (
    message_id VARCHAR(255) PRIMARY KEY,
    conversation_id VARCHAR(255) NOT NULL,
    role VARCHAR(50) NOT NULL, -- 'user' | 'assistant' | 'system'
    content TEXT NOT NULL,
    metadata JSONB, -- agent actions, tool calls, etc.
    embedding_id VARCHAR(255), -- reference to vector DB
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    FOREIGN KEY (conversation_id) REFERENCES conversations(conversation_id) ON DELETE CASCADE
);

CREATE INDEX idx_messages_conversation_id ON messages(conversation_id);
CREATE INDEX idx_messages_created_at ON messages(created_at DESC);
CREATE INDEX idx_messages_role ON messages(role);

-- Create agent_executions table to track AI agent runs
CREATE TABLE agent_executions (
    execution_id VARCHAR(255) PRIMARY KEY,
    agent_type VARCHAR(100) NOT NULL, -- 'document_parser' | 'underwriting' | 'memo' | 'chat'
    entity_id VARCHAR(255), -- deal_id, document_id, etc.
    entity_type VARCHAR(100), -- 'deal' | 'document' | 'conversation'
    input JSONB NOT NULL,
    output JSONB,
    status VARCHAR(50) NOT NULL, -- 'pending' | 'running' | 'completed' | 'failed'
    error TEXT,
    llm_provider VARCHAR(50), -- 'openai' | 'claude'
    model VARCHAR(100), -- 'gpt-4-turbo' | 'claude-3-5-sonnet'
    tokens_used INTEGER,
    execution_time_ms INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    completed_at TIMESTAMP
);

CREATE INDEX idx_agent_executions_entity ON agent_executions(entity_type, entity_id);
CREATE INDEX idx_agent_executions_status ON agent_executions(status);
CREATE INDEX idx_agent_executions_agent_type ON agent_executions(agent_type);
CREATE INDEX idx_agent_executions_created_at ON agent_executions(created_at DESC);

-- Create investor_memos table for generated memos
CREATE TABLE investor_memos (
    memo_id VARCHAR(255) PRIMARY KEY,
    deal_id VARCHAR(255) NOT NULL,
    title VARCHAR(500),
    content TEXT NOT NULL,
    sections JSONB, -- structured memo sections
    version INTEGER DEFAULT 1,
    status VARCHAR(50), -- 'draft' | 'final'
    created_by_agent BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    FOREIGN KEY (deal_id) REFERENCES deals(deal_id) ON DELETE CASCADE
);

CREATE INDEX idx_investor_memos_deal_id ON investor_memos(deal_id);
CREATE INDEX idx_investor_memos_status ON investor_memos(status);
CREATE INDEX idx_investor_memos_created_at ON investor_memos(created_at DESC);

-- Update documents table to track OCR and AI processing
ALTER TABLE documents 
ADD COLUMN ocr_completed_at TIMESTAMP,
ADD COLUMN fact_extraction_status VARCHAR(50) DEFAULT 'pending', -- 'pending' | 'processing' | 'completed' | 'failed'
ADD COLUMN fact_extraction_completed_at TIMESTAMP,
ADD COLUMN embedding_id VARCHAR(255); -- reference to vector DB

CREATE INDEX idx_documents_fact_extraction_status ON documents(fact_extraction_status);

-- Update facts table with confidence scores and extraction metadata
ALTER TABLE facts
ADD COLUMN confidence_score FLOAT CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0), -- 0.0 to 1.0
ADD COLUMN extraction_method VARCHAR(50) DEFAULT 'ai', -- 'ai' | 'manual' | 'rule_based'
ADD COLUMN reviewed_by_user BOOLEAN DEFAULT FALSE,
ADD COLUMN embedding_id VARCHAR(255); -- reference to vector DB for semantic search

CREATE INDEX idx_facts_confidence_score ON facts(confidence_score);
CREATE INDEX idx_facts_extraction_method ON facts(extraction_method);
CREATE INDEX idx_facts_reviewed_by_user ON facts(reviewed_by_user);

