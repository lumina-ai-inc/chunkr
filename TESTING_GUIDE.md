# AI Agent Testing Guide

## Prerequisites

1. **Environment Setup:**
   ```bash
   # Copy and configure environment variables
   cp .env.example .env
   # Add your API keys:
   # - OPENAI_API_KEY
   # - ANTHROPIC_API_KEY
   ```

2. **Start Services:**
   ```bash
   # Start Qdrant Vector DB
   cd docker
   docker-compose -f docker-compose.qdrant.yml up -d
   
   # Verify Qdrant is running
   curl http://localhost:6333/collections
   ```

3. **Run Migrations:**
   ```bash
   cd core
   diesel migration run
   ```

## Test 1: Basic Chat Flow (Landing Page)

### Objective: Test public chat with AI responses

1. **Start Frontend:**
   ```bash
   cd apps/web
   npm run dev
   ```

2. **Navigate to Landing Page:**
   - Open `http://localhost:5173`

3. **Test Chat:**
   - Type: "What can you tell me from a rent roll?"
   - **Expected:** AI-generated response explaining rent roll analysis
   - Type: "How do you calculate DSCR?"
   - **Expected:** Detailed explanation of DSCR calculation

4. **Verify:**
   - Responses are contextual and relevant
   - No errors in browser console
   - Messages appear in correct order

### Debugging:
```bash
# Check backend logs
tail -f core/logs/server.log

# Check if API endpoint is accessible
curl -X POST http://localhost:8000/api/v1/conversations/public/message \
  -H "Content-Type: application/json" \
  -d '{"content": "Hello"}'
```

## Test 2: Document Parsing & Fact Extraction

### Objective: Test AI-powered document parsing

1. **Start Backend:**
   ```bash
   cd core
   cargo run
   ```

2. **Login to Dashboard:**
   - Navigate to `http://localhost:5173/dashboard`
   - Sign in with your credentials

3. **Create New Deal:**
   - Click "+ New Deal"
   - Name: "Test Property Analysis"
   - Click "Create"

4. **Upload Document:**
   - Prepare a rent roll PDF or P&L statement
   - Click "Upload Documents"
   - Select your file
   - Click "Upload"

5. **Monitor Processing:**
   - Watch the document status change from "pending" → "processing" → "completed"
   - Navigate to "Deal Data Review" tab

6. **Verify Facts:**
   - Check that facts are extracted
   - Each fact should have:
     - ✅ Label (e.g., "Gross Scheduled Rent")
     - ✅ Value (e.g., "$250,000")
     - ✅ Unit (e.g., "USD/year")
     - ✅ Confidence score (0.0-1.0)
     - ✅ Status indicator (🟡 Needs Review, 🟢 Checked)

### Expected Facts for Rent Roll:
- Unit Count
- Occupancy Rate
- Gross Scheduled Rent
- Collected Rent

### Expected Facts for P&L:
- Gross Revenue
- Operating Expenses
- Net Operating Income (NOI)

### Debugging:
```bash
# Check agent execution logs
psql -d orin_db -c "SELECT * FROM agent_executions ORDER BY created_at DESC LIMIT 5;"

# Check extracted facts
psql -d orin_db -c "SELECT fact_type, label, value, confidence_score FROM facts WHERE deal_id = 'YOUR_DEAL_ID';"

# Check Qdrant embeddings
curl http://localhost:6333/collections/documents/points/scroll?limit=5
```

## Test 3: Underwriting Analysis

### Objective: Test underwriting calculations

1. **After Facts are Extracted:**
   - Go to "Deal Data Review" tab
   - Click "Verify All" to approve facts
   - Click "Run Underwriting"

2. **Navigate to Underwriting:**
   - Click "Underwriting Model" in left nav
   - Select your deal

3. **Verify Metrics:**
   - ✅ NOI (Net Operating Income)
   - ✅ DSCR (Debt Service Coverage Ratio)
   - ✅ Cash Flow After Debt
   - ✅ Cap Rate (if property value provided)
   - ✅ LTV (if mortgage data provided)

4. **Test Stress Scenarios:**
   - Adjust sliders:
     - Occupancy: -10%
     - Rent: -10%
     - Expenses: +10%
   - Verify metrics update in real-time

5. **Check Calculation Audit Trail:**
   - Each metric should show:
     - Formula used
     - Input values
     - Calculation steps
     - Result

### Debugging:
```bash
# Check underwriting results
psql -d orin_db -c "SELECT * FROM agent_executions WHERE agent_type = 'underwriting' ORDER BY created_at DESC LIMIT 1;"

# Verify facts used in calculation
curl http://localhost:8000/api/v1/deals/YOUR_DEAL_ID/facts
```

## Test 4: Authenticated Chat with Context (RAG)

### Objective: Test chat with deal context and vector search

1. **Start Conversation:**
   - In dashboard, open chat pane
   - Select a deal from dropdown

2. **Test Context-Aware Questions:**
   - Type: "What's the NOI for this property?"
   - **Expected:** AI retrieves NOI from your deal facts and responds with the specific value
   
   - Type: "Is the DSCR acceptable?"
   - **Expected:** AI analyzes the DSCR value and provides professional assessment
   
   - Type: "What are the risks with this deal?"
   - **Expected:** AI analyzes facts and provides risk assessment

3. **Verify Context Usage:**
   - Responses should reference specific numbers from your deal
   - AI should cite sources (e.g., "Based on your rent roll...")

### Debugging:
```bash
# Check conversation history
psql -d orin_db -c "SELECT role, content FROM messages WHERE conversation_id = 'YOUR_CONV_ID' ORDER BY created_at;"

# Check vector search results
# (Add logging in chat_orchestrator.rs to see retrieved context)
```

## Test 5: Vector Search (RAG)

### Objective: Verify semantic search is working

1. **Create Test Data:**
   - Upload 2-3 documents with different content
   - Ensure facts are extracted

2. **Test Search:**
   ```bash
   # Check Qdrant collections
   curl http://localhost:6333/collections
   
   # Check document embeddings
   curl http://localhost:6333/collections/documents/points/count
   
   # Check fact embeddings
   curl http://localhost:6333/collections/facts/points/count
   ```

3. **Test in Chat:**
   - Ask: "What documents mention NOI?"
   - **Expected:** AI finds and references relevant documents
   
   - Ask: "Show me all the rent information"
   - **Expected:** AI retrieves rent-related facts from vector DB

### Debugging:
```bash
# Search for specific embedding
curl -X POST http://localhost:6333/collections/facts/points/search \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.1, 0.2, ...],  # Your query embedding
    "limit": 5,
    "with_payload": true
  }'
```

## Common Issues & Solutions

### Issue 1: "OPENAI_API_KEY not set"
**Solution:**
```bash
# Add to .env file
OPENAI_API_KEY=sk-your-key-here
# Restart backend
```

### Issue 2: "Failed to connect to Qdrant"
**Solution:**
```bash
# Check if Qdrant is running
docker ps | grep qdrant

# Restart Qdrant
docker-compose -f docker/docker-compose.qdrant.yml restart

# Check logs
docker logs qdrant
```

### Issue 3: "No facts extracted from document"
**Possible Causes:**
- OCR failed (check document quality)
- Document type not recognized
- API rate limit reached

**Debug:**
```bash
# Check OCR output
psql -d orin_db -c "SELECT ocr_output FROM documents WHERE document_id = 'YOUR_DOC_ID';"

# Check agent execution errors
psql -d orin_db -c "SELECT error FROM agent_executions WHERE status = 'failed' ORDER BY created_at DESC LIMIT 5;"
```

### Issue 4: "Chat responses are slow"
**Possible Causes:**
- Vector search taking time
- Large conversation history
- API latency

**Optimize:**
- Limit conversation history to last 20 messages
- Use faster embedding model
- Add caching for frequent queries

### Issue 5: "Confidence scores are low"
**Causes:**
- Poor OCR quality
- Ambiguous document format
- Missing context

**Solutions:**
- Use higher quality scans
- Provide clearer document structure
- Adjust `FACT_EXTRACTION_MIN_CONFIDENCE` threshold

## Performance Benchmarks

### Expected Response Times:
- Public chat: < 3 seconds
- Authenticated chat with RAG: < 5 seconds
- Document parsing (10 pages): < 2 minutes
- Fact extraction: < 30 seconds
- Underwriting calculation: < 10 seconds

### Token Usage (Typical):
- Chat message: 500-1500 tokens
- Document parsing: 2000-5000 tokens per page
- Fact extraction: 1000-3000 tokens per document

## Success Criteria

✅ **Chat Flow:**
- [ ] Public chat responds within 3 seconds
- [ ] Responses are contextually relevant
- [ ] Session continuity works
- [ ] Login prompt appears after 2-3 exchanges

✅ **Document Parsing:**
- [ ] Documents upload successfully
- [ ] OCR completes without errors
- [ ] Facts extracted with confidence > 0.7
- [ ] Source citations include page numbers
- [ ] Embeddings stored in Qdrant

✅ **Underwriting:**
- [ ] Metrics calculate correctly
- [ ] Stress test sliders work
- [ ] Audit trail shows formulas
- [ ] Warnings appear for risky metrics

✅ **RAG (Context Retrieval):**
- [ ] Chat retrieves relevant facts
- [ ] Responses cite specific deal data
- [ ] Vector search returns < 1 second
- [ ] Context relevance score > 0.5

## Next Steps After Testing

1. **If all tests pass:**
   - Deploy to staging environment
   - Run load tests
   - Monitor API costs
   - Gather user feedback

2. **If tests fail:**
   - Check error logs
   - Verify API keys
   - Ensure all services are running
   - Review implementation summary

3. **Optimization:**
   - Add caching for frequent queries
   - Implement response streaming
   - Batch embedding operations
   - Add rate limiting

## Support

If you encounter issues:
1. Check logs in `core/logs/`
2. Review `AI_AGENT_IMPLEMENTATION_SUMMARY.md`
3. Verify environment variables
4. Ensure all services are running
5. Check API quotas and limits

---

**Happy Testing! 🚀**

