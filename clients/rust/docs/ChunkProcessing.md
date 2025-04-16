# ChunkProcessing

## Properties

Name | Type | Description | Notes
------------ | ------------- | ------------- | -------------
**ignore_headers_and_footers** | Option<**bool**> | Whether to ignore headers and footers in the chunking process. This is recommended as headers and footers break reading order across pages. | [optional][default to true]
**target_length** | Option<**i32**> | The target number of words in each chunk. If 0, each chunk will contain a single segment. | [optional][default to 512]
**tokenizer** | Option<[**models::ChunkProcessingTokenizer**](ChunkProcessing_tokenizer.md)> |  | [optional]

[[Back to Model list]](../README.md#documentation-for-models) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to README]](../README.md)


