use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ============================================================================
// REQUEST MODELS
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateContentRequest {
    /// Required. The content of the current conversation with the model.
    pub contents: Vec<Content>,
    /// Optional. A list of Tools the Model may use to generate the next response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tools: Option<Vec<Tool>>,
    /// Optional. Tool configuration for any Tool specified in the request.
    #[serde(skip_serializing_if = "Option::is_none", rename = "toolConfig")]
    pub tool_config: Option<ToolConfig>,
    /// Optional. A list of unique SafetySetting instances for blocking unsafe content.
    #[serde(skip_serializing_if = "Option::is_none", rename = "safetySettings")]
    pub safety_settings: Option<Vec<SafetySetting>>,
    /// Optional. Developer set system instruction(s). Currently, text only.
    #[serde(skip_serializing_if = "Option::is_none", rename = "systemInstruction")]
    pub system_instruction: Option<Content>,
    /// Optional. Configuration options for model generation and outputs.
    #[serde(skip_serializing_if = "Option::is_none", rename = "generationConfig")]
    pub generation_config: Option<GenerationConfig>,
    /// Optional. The name of the content cached to use as context to serve the prediction.
    #[serde(skip_serializing_if = "Option::is_none", rename = "cachedContent")]
    pub cached_content: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Content {
    /// Optional. The producer of the content. Must be either 'user' or 'model'.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    /// Ordered Parts that constitute a single message.
    pub parts: Vec<Part>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Part {
    Text {
        text: String,
    },
    InlineData {
        inline_data: Blob,
    },
    FileData {
        file_data: FileData,
    },
    FunctionCall {
        function_call: FunctionCall,
    },
    FunctionResponse {
        function_response: FunctionResponse,
    },
    CodeExecutionResult {
        code_execution_result: CodeExecutionResult,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Blob {
    /// The MIME type of the source data.
    pub mime_type: String,
    /// The base64-encoded bytes of the source data.
    pub data: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileData {
    /// The MIME type of the source data.
    pub mime_type: String,
    /// The URI of the source data.
    pub file_uri: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCall {
    /// Required. The name of the function to call.
    pub name: String,
    /// Optional. The function parameters and values in JSON object format.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub args: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionResponse {
    /// Required. The name of the function to call.
    pub name: String,
    /// Required. The function response in JSON object format.
    pub response: serde_json::Value,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExecutionResult {
    /// Outcome of the code execution.
    pub outcome: CodeExecutionOutcome,
    /// Optional. Contains stdout when code execution is successful, stderr or other description otherwise.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum CodeExecutionOutcome {
    #[serde(rename = "OUTCOME_UNSPECIFIED")]
    OutcomeUnspecified,
    #[serde(rename = "OUTCOME_OK")]
    OutcomeOk,
    #[serde(rename = "OUTCOME_FAILED")]
    OutcomeFailed,
    #[serde(rename = "OUTCOME_DEADLINE_EXCEEDED")]
    OutcomeDeadlineExceeded,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tool {
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "functionDeclarations"
    )]
    pub function_declarations: Option<Vec<FunctionDeclaration>>,
    #[serde(skip_serializing_if = "Option::is_none", rename = "codeExecution")]
    pub code_execution: Option<CodeExecution>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDeclaration {
    /// Required. The name of the function.
    pub name: String,
    /// Required. A brief description of the function.
    pub description: String,
    /// Optional. Describes the parameters to this function.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parameters: Option<Schema>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeExecution {}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolConfig {
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "functionCallingConfig"
    )]
    pub function_calling_config: Option<FunctionCallingConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallingConfig {
    /// Optional. Specifies the mode in which function calling should execute.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mode: Option<FunctionCallingMode>,
    /// Optional. A set of function names that, when provided, limits the functions the model will call.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "allowedFunctionNames"
    )]
    pub allowed_function_names: Option<Vec<String>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FunctionCallingMode {
    #[serde(rename = "MODE_UNSPECIFIED")]
    ModeUnspecified,
    #[serde(rename = "AUTO")]
    Auto,
    #[serde(rename = "ANY")]
    Any,
    #[serde(rename = "NONE")]
    None,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetySetting {
    /// Required. The category for this setting.
    pub category: HarmCategory,
    /// Required. Controls the probability threshold at which harm is blocked.
    pub threshold: HarmBlockThreshold,
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct GenerationConfig {
    /// Optional. The set of character sequences that will stop output generation.
    #[serde(skip_serializing_if = "Option::is_none", rename = "stopSequences")]
    pub stop_sequences: Option<Vec<String>>,
    /// Optional. MIME type of the generated candidate text.
    #[serde(skip_serializing_if = "Option::is_none", rename = "responseMimeType")]
    pub response_mime_type: Option<String>,
    /// Optional. Output schema of the generated candidate text.
    #[serde(skip_serializing_if = "Option::is_none", rename = "responseSchema")]
    pub response_schema: Option<Schema>,
    /// Optional. The requested modalities of the response.
    #[serde(skip_serializing_if = "Option::is_none", rename = "responseModalities")]
    pub response_modalities: Option<Vec<Modality>>,
    /// Optional. Number of generated responses to return.
    #[serde(skip_serializing_if = "Option::is_none", rename = "candidateCount")]
    pub candidate_count: Option<i32>,
    /// Optional. The maximum number of tokens to include in a response candidate.
    #[serde(skip_serializing_if = "Option::is_none", rename = "maxOutputTokens")]
    pub max_output_tokens: Option<i32>,
    /// Optional. Controls the randomness of the output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f64>,
    /// Optional. The maximum cumulative probability of tokens to consider when sampling.
    #[serde(skip_serializing_if = "Option::is_none", rename = "topP")]
    pub top_p: Option<f64>,
    /// Optional. The maximum number of tokens to consider when sampling.
    #[serde(skip_serializing_if = "Option::is_none", rename = "topK")]
    pub top_k: Option<i32>,
    /// Optional. Seed used in decoding.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seed: Option<i32>,
    /// Optional. Presence penalty applied to the next token's logprobs.
    #[serde(skip_serializing_if = "Option::is_none", rename = "presencePenalty")]
    pub presence_penalty: Option<f64>,
    /// Optional. Frequency penalty applied to the next token's logprobs.
    #[serde(skip_serializing_if = "Option::is_none", rename = "frequencyPenalty")]
    pub frequency_penalty: Option<f64>,
    /// Optional. If true, export the logprobs results in response.
    #[serde(skip_serializing_if = "Option::is_none", rename = "responseLogprobs")]
    pub response_logprobs: Option<bool>,
    /// Optional. Number of top logprobs to return at each decoding step.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub logprobs: Option<i32>,
    /// Optional. Enables enhanced civic answers.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "enableEnhancedCivicAnswers"
    )]
    pub enable_enhanced_civic_answers: Option<bool>,
    /// Optional. The speech generation config.
    #[serde(skip_serializing_if = "Option::is_none", rename = "speechConfig")]
    pub speech_config: Option<SpeechConfig>,
    /// Optional. Config for thinking features.
    #[serde(skip_serializing_if = "Option::is_none", rename = "thinkingConfig")]
    pub thinking_config: Option<ThinkingConfig>,
    /// Optional. Media resolution specified will be used.
    #[serde(skip_serializing_if = "Option::is_none", rename = "mediaResolution")]
    pub media_resolution: Option<MediaResolution>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Schema {
    /// Data type of the schema.
    #[serde(rename = "type")]
    pub schema_type: SchemaType,
    /// Optional. The format of the data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub format: Option<String>,
    /// Optional. A brief description of the parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    /// Optional. Indicates if the value may be null.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub nullable: Option<bool>,
    /// Optional. Possible values of the element of Type.STRING with enum format.
    #[serde(skip_serializing_if = "Option::is_none", rename = "enum")]
    pub enum_values: Option<Vec<String>>,
    /// Optional. Schema of the elements of Type.ARRAY.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub items: Option<Box<Schema>>,
    /// Optional. Properties of Type.OBJECT.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub properties: Option<HashMap<String, Schema>>,
    /// Optional. Required properties of Type.OBJECT.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub required: Option<Vec<String>>,
    /// Optional. Minimum value of the Type.INTEGER and Type.NUMBER
    #[serde(skip_serializing_if = "Option::is_none")]
    pub minimum: Option<f64>,
    /// Optional. Maximum value of the Type.INTEGER and Type.NUMBER
    #[serde(skip_serializing_if = "Option::is_none")]
    pub maximum: Option<f64>,
    /// Optional. Minimum number of the elements for Type.ARRAY.
    #[serde(skip_serializing_if = "Option::is_none", rename = "minItems")]
    pub min_items: Option<i32>,
    /// Optional. Maximum number of the elements for Type.ARRAY.
    #[serde(skip_serializing_if = "Option::is_none", rename = "maxItems")]
    pub max_items: Option<i32>,
    /// Optional. Minimum length of the Type.STRING
    #[serde(skip_serializing_if = "Option::is_none", rename = "minLength")]
    pub min_length: Option<i32>,
    /// Optional. Maximum length of the Type.STRING
    #[serde(skip_serializing_if = "Option::is_none", rename = "maxLength")]
    pub max_length: Option<i32>,
    /// Optional. Pattern of the Type.STRING to restrict a string to a regular expression.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pattern: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum SchemaType {
    #[serde(rename = "TYPE_UNSPECIFIED")]
    TypeUnspecified,
    #[serde(rename = "STRING")]
    String,
    #[serde(rename = "NUMBER")]
    Number,
    #[serde(rename = "INTEGER")]
    Integer,
    #[serde(rename = "BOOLEAN")]
    Boolean,
    #[serde(rename = "ARRAY")]
    Array,
    #[serde(rename = "OBJECT")]
    Object,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeechConfig {
    /// The configuration in case of single-voice output.
    #[serde(skip_serializing_if = "Option::is_none", rename = "voiceConfig")]
    pub voice_config: Option<VoiceConfig>,
    /// Optional. The configuration for the multi-speaker setup.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "multiSpeakerVoiceConfig"
    )]
    pub multi_speaker_voice_config: Option<MultiSpeakerVoiceConfig>,
    /// Optional. Language code for speech synthesis.
    #[serde(skip_serializing_if = "Option::is_none", rename = "languageCode")]
    pub language_code: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VoiceConfig {
    /// The configuration for the prebuilt voice to use.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "prebuiltVoiceConfig"
    )]
    pub prebuilt_voice_config: Option<PrebuiltVoiceConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrebuiltVoiceConfig {
    /// The name of the preset voice to use.
    #[serde(rename = "voiceName")]
    pub voice_name: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultiSpeakerVoiceConfig {
    /// Required. All the enabled speaker voices.
    #[serde(rename = "speakerVoiceConfigs")]
    pub speaker_voice_configs: Vec<SpeakerVoiceConfig>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpeakerVoiceConfig {
    /// Required. The name of the speaker to use.
    pub speaker: String,
    /// Required. The configuration for the voice to use.
    #[serde(rename = "voiceConfig")]
    pub voice_config: VoiceConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThinkingConfig {
    /// Indicates whether to include thoughts in the response.
    #[serde(skip_serializing_if = "Option::is_none", rename = "includeThoughts")]
    pub include_thoughts: Option<bool>,
    /// The number of thoughts tokens that the model should generate.
    #[serde(skip_serializing_if = "Option::is_none", rename = "thinkingBudget")]
    pub thinking_budget: Option<i32>,
}

// ============================================================================
// RESPONSE MODELS
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GenerateContentResponse {
    /// Candidate responses from the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidates: Option<Vec<Candidate>>,
    /// Returns the prompt's feedback related to the content filters.
    #[serde(skip_serializing_if = "Option::is_none", rename = "promptFeedback")]
    pub prompt_feedback: Option<PromptFeedback>,
    /// Output only. Metadata on the generation requests' token usage.
    #[serde(skip_serializing_if = "Option::is_none", rename = "usageMetadata")]
    pub usage_metadata: Option<UsageMetadata>,
    /// Output only. The model version used to generate the response.
    #[serde(skip_serializing_if = "Option::is_none", rename = "modelVersion")]
    pub model_version: Option<String>,
    /// Output only. responseId is used to identify each response.
    #[serde(skip_serializing_if = "Option::is_none", rename = "responseId")]
    pub response_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candidate {
    /// Output only. Generated content returned from the model.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Content>,
    /// Optional. Output only. The reason why the model stopped generating tokens.
    #[serde(skip_serializing_if = "Option::is_none", rename = "finishReason")]
    pub finish_reason: Option<FinishReason>,
    /// List of ratings for the safety of a response candidate.
    #[serde(skip_serializing_if = "Option::is_none", rename = "safetyRatings")]
    pub safety_ratings: Option<Vec<SafetyRating>>,
    /// Output only. Citation information for model-generated candidate.
    #[serde(skip_serializing_if = "Option::is_none", rename = "citationMetadata")]
    pub citation_metadata: Option<CitationMetadata>,
    /// Output only. Token count for this candidate.
    #[serde(skip_serializing_if = "Option::is_none", rename = "tokenCount")]
    pub token_count: Option<i32>,
    /// Output only. Attribution information for sources that contributed to a grounded answer.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "groundingAttributions"
    )]
    pub grounding_attributions: Option<Vec<GroundingAttribution>>,
    /// Output only. Grounding metadata for the candidate.
    #[serde(skip_serializing_if = "Option::is_none", rename = "groundingMetadata")]
    pub grounding_metadata: Option<GroundingMetadata>,
    /// Output only. Average log probability score of the candidate.
    #[serde(skip_serializing_if = "Option::is_none", rename = "avgLogprobs")]
    pub avg_logprobs: Option<f64>,
    /// Output only. Log-likelihood scores for the response tokens and top tokens
    #[serde(skip_serializing_if = "Option::is_none", rename = "logprobsResult")]
    pub logprobs_result: Option<LogprobsResult>,
    /// Output only. Index of the candidate in the list of response candidates.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub index: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptFeedback {
    /// Optional. If set, the prompt was blocked and no candidates are returned.
    #[serde(skip_serializing_if = "Option::is_none", rename = "blockReason")]
    pub block_reason: Option<BlockReason>,
    /// Ratings for safety of the prompt.
    #[serde(skip_serializing_if = "Option::is_none", rename = "safetyRatings")]
    pub safety_ratings: Option<Vec<SafetyRating>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageMetadata {
    /// Number of tokens in the prompt.
    #[serde(skip_serializing_if = "Option::is_none", rename = "promptTokenCount")]
    pub prompt_token_count: Option<i32>,
    /// Number of tokens in the cached part of the prompt.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "cachedContentTokenCount"
    )]
    pub cached_content_token_count: Option<i32>,
    /// Total number of tokens across all the generated response candidates.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "candidatesTokenCount"
    )]
    pub candidates_token_count: Option<i32>,
    /// Output only. Number of tokens present in tool-use prompt(s).
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "toolUsePromptTokenCount"
    )]
    pub tool_use_prompt_token_count: Option<i32>,
    /// Output only. Number of tokens of thoughts for thinking models.
    #[serde(skip_serializing_if = "Option::is_none", rename = "thoughtsTokenCount")]
    pub thoughts_token_count: Option<i32>,
    /// Total token count for the generation request.
    #[serde(skip_serializing_if = "Option::is_none", rename = "totalTokenCount")]
    pub total_token_count: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitationMetadata {
    /// Citations to sources for a specific response.
    #[serde(skip_serializing_if = "Option::is_none", rename = "citationSources")]
    pub citation_sources: Option<Vec<CitationSource>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CitationSource {
    /// Optional. Start of segment of the response that is attributed to this source.
    #[serde(skip_serializing_if = "Option::is_none", rename = "startIndex")]
    pub start_index: Option<i32>,
    /// Optional. End of the attributed segment, exclusive.
    #[serde(skip_serializing_if = "Option::is_none", rename = "endIndex")]
    pub end_index: Option<i32>,
    /// Optional. URI that is attributed as a source for a portion of the text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
    /// Optional. License for the GitHub project that is attributed as a source for segment.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub license: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundingAttribution {
    /// Output only. Identifier for the source contributing to this attribution.
    #[serde(skip_serializing_if = "Option::is_none", rename = "sourceId")]
    pub source_id: Option<AttributionSourceId>,
    /// Grounding source content that makes up this attribution.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<Content>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttributionSourceId {
    /// Identifier for an inline passage.
    #[serde(skip_serializing_if = "Option::is_none", rename = "groundingPassage")]
    pub grounding_passage: Option<GroundingPassageId>,
    /// Identifier for a Chunk fetched via Semantic Retriever.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "semanticRetrieverChunk"
    )]
    pub semantic_retriever_chunk: Option<SemanticRetrieverChunk>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundingPassageId {
    /// Output only. ID of the passage matching the GenerateAnswerRequest's GroundingPassage.id.
    #[serde(skip_serializing_if = "Option::is_none", rename = "passageId")]
    pub passage_id: Option<String>,
    /// Output only. Index of the part within the GenerateAnswerRequest's GroundingPassage.content.
    #[serde(skip_serializing_if = "Option::is_none", rename = "partIndex")]
    pub part_index: Option<i32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticRetrieverChunk {
    /// Output only. Name of the source matching the request's SemanticRetrieverConfig.source.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// Output only. Name of the Chunk containing the attributed text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundingMetadata {
    /// List of supporting references retrieved from specified grounding source.
    #[serde(skip_serializing_if = "Option::is_none", rename = "groundingChunks")]
    pub grounding_chunks: Option<Vec<GroundingChunk>>,
    /// List of grounding support.
    #[serde(skip_serializing_if = "Option::is_none", rename = "groundingSupports")]
    pub grounding_supports: Option<Vec<GroundingSupport>>,
    /// Web search queries for the following-up web search.
    #[serde(skip_serializing_if = "Option::is_none", rename = "webSearchQueries")]
    pub web_search_queries: Option<Vec<String>>,
    /// Optional. Google search entry for the following-up web searches.
    #[serde(skip_serializing_if = "Option::is_none", rename = "searchEntryPoint")]
    pub search_entry_point: Option<SearchEntryPoint>,
    /// Metadata related to retrieval in the grounding flow.
    #[serde(skip_serializing_if = "Option::is_none", rename = "retrievalMetadata")]
    pub retrieval_metadata: Option<RetrievalMetadata>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundingChunk {
    /// Grounding chunk from the web.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub web: Option<Web>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Web {
    /// URI reference of the chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub uri: Option<String>,
    /// Title of the chunk.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GroundingSupport {
    /// A list of indices specifying the citations associated with the claim.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "groundingChunkIndices"
    )]
    pub grounding_chunk_indices: Option<Vec<i32>>,
    /// Confidence score of the support references.
    #[serde(skip_serializing_if = "Option::is_none", rename = "confidenceScores")]
    pub confidence_scores: Option<Vec<f64>>,
    /// Segment of the content this support belongs to.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub segment: Option<Segment>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    /// Output only. The index of a Part object within its parent Content object.
    #[serde(skip_serializing_if = "Option::is_none", rename = "partIndex")]
    pub part_index: Option<i32>,
    /// Output only. Start index in the given Part, measured in bytes.
    #[serde(skip_serializing_if = "Option::is_none", rename = "startIndex")]
    pub start_index: Option<i32>,
    /// Output only. End index in the given Part, measured in bytes.
    #[serde(skip_serializing_if = "Option::is_none", rename = "endIndex")]
    pub end_index: Option<i32>,
    /// Output only. The text corresponding to the segment from the response.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchEntryPoint {
    /// Optional. Web content snippet that can be embedded in a web page or an app webview.
    #[serde(skip_serializing_if = "Option::is_none", rename = "renderedContent")]
    pub rendered_content: Option<String>,
    /// Optional. Base64 encoded JSON representing array of <search term, search url> tuple.
    #[serde(skip_serializing_if = "Option::is_none", rename = "sdkBlob")]
    pub sdk_blob: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetrievalMetadata {
    /// Optional. Score indicating how likely information from google search could help answer the prompt.
    #[serde(
        skip_serializing_if = "Option::is_none",
        rename = "googleSearchDynamicRetrievalScore"
    )]
    pub google_search_dynamic_retrieval_score: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogprobsResult {
    /// Length = total number of decoding steps.
    #[serde(skip_serializing_if = "Option::is_none", rename = "topCandidates")]
    pub top_candidates: Option<Vec<TopCandidates>>,
    /// Length = total number of decoding steps.
    #[serde(skip_serializing_if = "Option::is_none", rename = "chosenCandidates")]
    pub chosen_candidates: Option<Vec<LogprobsCandidate>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TopCandidates {
    /// Sorted by log probability in descending order.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub candidates: Option<Vec<LogprobsCandidate>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogprobsCandidate {
    /// The candidate's token string value.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token: Option<String>,
    /// The candidate's token id value.
    #[serde(skip_serializing_if = "Option::is_none", rename = "tokenId")]
    pub token_id: Option<i32>,
    /// The candidate's log probability.
    #[serde(skip_serializing_if = "Option::is_none", rename = "logProbability")]
    pub log_probability: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SafetyRating {
    /// Required. The category for this rating.
    pub category: HarmCategory,
    /// Required. The probability of harm for this content.
    pub probability: HarmProbability,
    /// Was this content blocked because of this rating?
    #[serde(skip_serializing_if = "Option::is_none")]
    pub blocked: Option<bool>,
}

// ============================================================================
// ENUMS
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum Modality {
    #[serde(rename = "MODALITY_UNSPECIFIED")]
    ModalityUnspecified,
    #[serde(rename = "TEXT")]
    Text,
    #[serde(rename = "IMAGE")]
    Image,
    #[serde(rename = "AUDIO")]
    Audio,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum MediaResolution {
    #[serde(rename = "MEDIA_RESOLUTION_UNSPECIFIED")]
    MediaResolutionUnspecified,
    #[serde(rename = "MEDIA_RESOLUTION_LOW")]
    MediaResolutionLow,
    #[serde(rename = "MEDIA_RESOLUTION_MEDIUM")]
    MediaResolutionMedium,
    #[serde(rename = "MEDIA_RESOLUTION_HIGH")]
    MediaResolutionHigh,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum FinishReason {
    #[serde(rename = "FINISH_REASON_UNSPECIFIED")]
    FinishReasonUnspecified,
    #[serde(rename = "STOP")]
    Stop,
    #[serde(rename = "MAX_TOKENS")]
    MaxTokens,
    #[serde(rename = "SAFETY")]
    Safety,
    #[serde(rename = "RECITATION")]
    Recitation,
    #[serde(rename = "LANGUAGE")]
    Language,
    #[serde(rename = "OTHER")]
    Other,
    #[serde(rename = "BLOCKLIST")]
    Blocklist,
    #[serde(rename = "PROHIBITED_CONTENT")]
    ProhibitedContent,
    #[serde(rename = "SPII")]
    Spii,
    #[serde(rename = "MALFORMED_FUNCTION_CALL")]
    MalformedFunctionCall,
    #[serde(rename = "IMAGE_SAFETY")]
    ImageSafety,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum BlockReason {
    #[serde(rename = "BLOCK_REASON_UNSPECIFIED")]
    BlockReasonUnspecified,
    #[serde(rename = "SAFETY")]
    Safety,
    #[serde(rename = "OTHER")]
    Other,
    #[serde(rename = "BLOCKLIST")]
    Blocklist,
    #[serde(rename = "PROHIBITED_CONTENT")]
    ProhibitedContent,
    #[serde(rename = "IMAGE_SAFETY")]
    ImageSafety,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum HarmCategory {
    #[serde(rename = "HARM_CATEGORY_UNSPECIFIED")]
    HarmCategoryUnspecified,
    #[serde(rename = "HARM_CATEGORY_DEROGATORY")]
    HarmCategoryDerogatory,
    #[serde(rename = "HARM_CATEGORY_TOXICITY")]
    HarmCategoryToxicity,
    #[serde(rename = "HARM_CATEGORY_VIOLENCE")]
    HarmCategoryViolence,
    #[serde(rename = "HARM_CATEGORY_SEXUAL")]
    HarmCategorySexual,
    #[serde(rename = "HARM_CATEGORY_MEDICAL")]
    HarmCategoryMedical,
    #[serde(rename = "HARM_CATEGORY_DANGEROUS")]
    HarmCategoryDangerous,
    #[serde(rename = "HARM_CATEGORY_HARASSMENT")]
    HarmCategoryHarassment,
    #[serde(rename = "HARM_CATEGORY_HATE_SPEECH")]
    HarmCategoryHateSpeech,
    #[serde(rename = "HARM_CATEGORY_SEXUALLY_EXPLICIT")]
    HarmCategorySexuallyExplicit,
    #[serde(rename = "HARM_CATEGORY_DANGEROUS_CONTENT")]
    HarmCategoryDangerousContent,
    #[serde(rename = "HARM_CATEGORY_CIVIC_INTEGRITY")]
    HarmCategoryCivicIntegrity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum HarmProbability {
    #[serde(rename = "HARM_PROBABILITY_UNSPECIFIED")]
    HarmProbabilityUnspecified,
    #[serde(rename = "NEGLIGIBLE")]
    Negligible,
    #[serde(rename = "LOW")]
    Low,
    #[serde(rename = "MEDIUM")]
    Medium,
    #[serde(rename = "HIGH")]
    High,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum HarmBlockThreshold {
    #[serde(rename = "HARM_BLOCK_THRESHOLD_UNSPECIFIED")]
    HarmBlockThresholdUnspecified,
    #[serde(rename = "BLOCK_LOW_AND_ABOVE")]
    BlockLowAndAbove,
    #[serde(rename = "BLOCK_MEDIUM_AND_ABOVE")]
    BlockMediumAndAbove,
    #[serde(rename = "BLOCK_ONLY_HIGH")]
    BlockOnlyHigh,
    #[serde(rename = "BLOCK_NONE")]
    BlockNone,
    #[serde(rename = "OFF")]
    Off,
}

// ============================================================================
// HELPER IMPLEMENTATIONS
// ============================================================================

impl GenerateContentRequest {
    /// Create a simple text generation request
    pub fn new_text_request(text: &str) -> Self {
        Self {
            contents: vec![Content {
                role: Some("user".to_string()),
                parts: vec![Part::Text {
                    text: text.to_string(),
                }],
            }],
            tools: None,
            tool_config: None,
            safety_settings: None,
            system_instruction: None,
            generation_config: None,
            cached_content: None,
        }
    }

    /// Create a request with JSON mode
    pub fn new_json_request(text: &str, schema: Schema) -> Self {
        Self {
            contents: vec![Content {
                role: Some("user".to_string()),
                parts: vec![Part::Text {
                    text: text.to_string(),
                }],
            }],
            tools: None,
            tool_config: None,
            safety_settings: None,
            system_instruction: None,
            generation_config: Some(GenerationConfig {
                response_mime_type: Some("application/json".to_string()),
                response_schema: Some(schema),
                ..Default::default()
            }),
            cached_content: None,
        }
    }
}

impl GenerateContentResponse {
    /// Get the first candidate's text content, if available
    pub fn get_text(&self) -> Option<String> {
        self.candidates
            .as_ref()?
            .first()?
            .content
            .as_ref()?
            .parts
            .iter()
            .find_map(|part| match part {
                Part::Text { text } => Some(text.clone()),
                _ => None,
            })
    }
}

/// Converts a schemars-generated JSON schema to Gemini-compatible format
///
/// This function converts standard JSON schemas to Gemini's Schema format:
/// - Maps JSON schema types to Gemini SchemaType enum
/// - Converts properties and nested schemas recursively
/// - Preserves descriptions from doc comments
/// - Handles arrays, objects, and primitive types appropriately
pub fn convert_schema_to_genai_format(
    schema_json: serde_json::Value,
) -> Result<Schema, Box<dyn std::error::Error + Send + Sync>> {
    let obj = schema_json.as_object().ok_or("Schema must be an object")?;

    // Determine the schema type
    let schema_type = match obj.get("type").and_then(|t| t.as_str()) {
        Some("string") => SchemaType::String,
        Some("number") => SchemaType::Number,
        Some("integer") => SchemaType::Integer,
        Some("boolean") => SchemaType::Boolean,
        Some("array") => SchemaType::Array,
        Some("object") => SchemaType::Object,
        _ => SchemaType::TypeUnspecified,
    };

    // Extract description
    let description = obj
        .get("description")
        .and_then(|d| d.as_str())
        .map(|s| s.to_string());

    // Extract nullable
    let nullable = obj.get("nullable").and_then(|n| n.as_bool());

    // Extract enum values
    let enum_values = obj.get("enum").and_then(|e| e.as_array()).map(|arr| {
        arr.iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect()
    });

    // Handle array items
    let items = if schema_type == SchemaType::Array {
        obj.get("items")
            .map(|items_schema| convert_schema_to_genai_format(items_schema.clone()))
            .transpose()?
            .map(Box::new)
    } else {
        None
    };

    // Handle object properties
    let properties = if schema_type == SchemaType::Object {
        obj.get("properties")
            .and_then(|p| p.as_object())
            .map(|props| {
                let mut result = HashMap::new();
                for (key, value) in props {
                    if let Ok(schema) = convert_schema_to_genai_format(value.clone()) {
                        result.insert(key.clone(), schema);
                    }
                }
                result
            })
    } else {
        None
    };

    // Handle required fields
    let required = obj.get("required").and_then(|r| r.as_array()).map(|arr| {
        arr.iter()
            .filter_map(|v| v.as_str().map(|s| s.to_string()))
            .collect()
    });

    // Extract numeric constraints
    let minimum = obj.get("minimum").and_then(|m| m.as_f64());
    let maximum = obj.get("maximum").and_then(|m| m.as_f64());

    // Extract array constraints
    let min_items = obj
        .get("minItems")
        .and_then(|m| m.as_i64())
        .map(|i| i as i32);
    let max_items = obj
        .get("maxItems")
        .and_then(|m| m.as_i64())
        .map(|i| i as i32);

    // Extract string constraints
    let min_length = obj
        .get("minLength")
        .and_then(|m| m.as_i64())
        .map(|i| i as i32);
    let max_length = obj
        .get("maxLength")
        .and_then(|m| m.as_i64())
        .map(|i| i as i32);
    let pattern = obj
        .get("pattern")
        .and_then(|p| p.as_str())
        .map(|s| s.to_string());

    Ok(Schema {
        schema_type,
        format: None, // Gemini doesn't use format field the same way
        description,
        nullable,
        enum_values,
        items,
        properties,
        required,
        minimum,
        maximum,
        min_items,
        max_items,
        min_length,
        max_length,
        pattern,
    })
}
