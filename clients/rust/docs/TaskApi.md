# \TaskApi

All URIs are relative to *https://api.chunkr.ai*

Method | HTTP request | Description
------------- | ------------- | -------------
[**cancel_task_route**](TaskApi.md#cancel_task_route) | **GET** /api/v1/task/{task_id}/cancel | Cancel Task
[**create_task_route**](TaskApi.md#create_task_route) | **POST** /api/v1/task/parse | Create Task
[**delete_task_route**](TaskApi.md#delete_task_route) | **DELETE** /api/v1/task/{task_id} | Delete Task
[**get_task_route**](TaskApi.md#get_task_route) | **GET** /api/v1/task/{task_id} | Get Task
[**update_task_route**](TaskApi.md#update_task_route) | **PATCH** /api/v1/task/{task_id}/parse | Update Task



## cancel_task_route

> cancel_task_route(task_id)
Cancel Task

Cancel a task that hasn't started processing yet: - For new tasks: Status will be updated to `Cancelled` - For updating tasks: Task will revert to the previous state  Requirements: - Task must have status `Starting`

### Parameters


Name | Type | Description  | Required | Notes
------------- | ------------- | ------------- | ------------- | -------------
**task_id** | Option<**String**> | Id of the task to cancel | [required] |

### Return type

 (empty response body)

### Authorization

[api_key](../README.md#api_key)

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: text/plain

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)


## create_task_route

> models::TaskResponse create_task_route(create_form)
Create Task

Queues a document for processing and returns a TaskResponse containing: - Task ID for status polling - Initial configuration - File metadata - Processing status - Creation timestamp - Presigned URLs for file access  The returned task will typically be in a `Starting` or `Processing` state. Use the `GET /task/{task_id}` endpoint to poll for completion.

### Parameters


Name | Type | Description  | Required | Notes
------------- | ------------- | ------------- | ------------- | -------------
**create_form** | [**CreateForm**](CreateForm.md) | JSON request to create a task | [required] |

### Return type

[**models::TaskResponse**](TaskResponse.md)

### Authorization

[api_key](../README.md#api_key)

### HTTP request headers

- **Content-Type**: application/json
- **Accept**: application/json, text/plain

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)


## delete_task_route

> delete_task_route(task_id)
Delete Task

Delete a task by its ID.  Requirements: - Task must have status `Succeeded` or `Failed`

### Parameters


Name | Type | Description  | Required | Notes
------------- | ------------- | ------------- | ------------- | -------------
**task_id** | Option<**String**> | Id of the task to delete | [required] |

### Return type

 (empty response body)

### Authorization

[api_key](../README.md#api_key)

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: text/plain

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)


## get_task_route

> models::TaskResponse get_task_route(task_id, base64_urls, include_chunks)
Get Task

Retrieves detailed information about a task by its ID, including: - Processing status - Task configuration - Output data (if processing is complete) - File metadata (name, page count) - Timestamps (created, started, finished) - Presigned URLs for accessing files  This endpoint can be used to: 1. Poll the task status during processing 2. Retrieve the final output once processing is complete 3. Access task metadata and configuration

### Parameters


Name | Type | Description  | Required | Notes
------------- | ------------- | ------------- | ------------- | -------------
**task_id** | Option<**String**> | Id of the task to retrieve | [required] |
**base64_urls** | Option<**bool**> | Whether to return base64 encoded URLs. If false, the URLs will be returned as presigned URLs. |  |
**include_chunks** | Option<**bool**> | Whether to include chunks in the output response |  |

### Return type

[**models::TaskResponse**](TaskResponse.md)

### Authorization

[api_key](../README.md#api_key)

### HTTP request headers

- **Content-Type**: Not defined
- **Accept**: application/json, text/plain

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)


## update_task_route

> models::TaskResponse update_task_route(task_id, update_form)
Update Task

Updates an existing task's configuration and reprocesses the document. The original configuration will be used for all values that are not provided in the update.  Requirements: - Task must have status `Succeeded` or `Failed` - New configuration must be different from the current one  The returned task will typically be in a `Starting` or `Processing` state. Use the `GET /task/{task_id}` endpoint to poll for completion.

### Parameters


Name | Type | Description  | Required | Notes
------------- | ------------- | ------------- | ------------- | -------------
**task_id** | **String** |  | [required] |
**update_form** | [**UpdateForm**](UpdateForm.md) | JSON request to update an task | [required] |

### Return type

[**models::TaskResponse**](TaskResponse.md)

### Authorization

[api_key](../README.md#api_key)

### HTTP request headers

- **Content-Type**: application/json
- **Accept**: application/json, text/plain

[[Back to top]](#) [[Back to API list]](../README.md#documentation-for-api-endpoints) [[Back to Model list]](../README.md#documentation-for-models) [[Back to README]](../README.md)

