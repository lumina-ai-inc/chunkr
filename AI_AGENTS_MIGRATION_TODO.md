# AI Agents Migration TODO

## Status: Temporarily Disabled

The AI agent modules (`chat_orchestrator`, `document_parser`) have been temporarily disabled during the backend build because they need to be migrated from Diesel ORM to raw tokio-postgres SQL queries.

### Why?

The codebase uses `tokio-postgres` with raw SQL queries, not Diesel ORM. The agents were initially written with Diesel `.await` queries which don't work with the current database setup.

### What's Disabled?

1. **`core/src/agents/` module** - Commented out in `lib.rs`
2. **`core/src/routes/deal.rs`** - All deal endpoints (uses Diesel)
3. **Chat orchestrator integration** - Conversation endpoints return placeholder responses
4. **Document parser agent** - Not triggered automatically

### What Still Works?

✅ All existing endpoints (user, tasks, etc.)
✅ Conversation API endpoints (placeholder responses)
✅ Frontend (falls back to template responses for chat)
✅ Original task/document processing pipeline

### What's Temporarily Unavailable?

❌ Deal creation and management (new feature - uses Diesel)
❌ Document upload to deals (new feature - uses Diesel)
❌ Fact extraction (new feature - uses Diesel)
❌ Underwriting calculations (new feature - uses Diesel)
❌ AI chat with RAG (new feature - uses Diesel)

### To Re-enable (Future Task):

1. **Convert agent queries to raw SQL:**
   - `chat_orchestrator.rs`: Lines with `diesel::` calls
   - `document_parser.rs`: Lines with `diesel::` calls
   - `common.rs`: `log_execution` function
   
2. **Pattern to follow:**
   ```rust
   // Instead of:
   diesel::insert_into(table).values(&data).execute(&mut client).await?;
   
   // Use:
   client.execute(
       "INSERT INTO table (col1, col2) VALUES ($1, $2)",
       &[&value1, &value2]
   ).await?;
   ```

3. **Re-enable in `lib.rs`:**
   ```rust
   pub mod agents; // Uncomment this line
   ```

4. **Update conversation routes** to use real `ChatOrchestratorAgent` instead of placeholder

### Current Priority:

Get the backend building and running first. The AI agents can be properly migrated in a follow-up session.

---

**Created:** 2026-01-05
**Reason:** Build error - diesel queries incompatible with tokio-postgres setup
**Impact:** Chat uses fallback templates; agents don't run automatically

