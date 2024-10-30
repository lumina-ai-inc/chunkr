model=BAAI/bge-reranker-large
revision=refs/pr/5

text-embeddings-router --model-id $model --revision $revision --port 8080