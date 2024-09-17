curl -X 'POST' \
    'http://localhost:3000/convert_to_img' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F 'file=@/Users/akhileshsharma/Documents/Lumina/chunk-my-docs/services/task/input/1c749c35-cc15-56b2-ade5-010fbf1a9778.pdf' \
    -F 'density=150' \
    -o output/output.json