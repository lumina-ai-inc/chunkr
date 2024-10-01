curl -X 'POST' \
    'http://localhost:3000/convert_to_img' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F 'file=@/Users/akhileshsharma/Documents/Lumina/chunk-my-docs/services/task/input/1c749c35-cc15-56b2-ade5-010fbf1a9778.pdf' \
    -F 'density=150' \
    -o output/output.json

uv run pip install paddlepaddle-gpu
uv add paddlepaddle-gpu

# [project.optional-dependencies]
# cpu = ["paddlepaddle>=2.6.2"]
# gpu = ["paddlepaddle-gpu>=2.6.2,<2.7.0"]

uv pip install .[linux]
uv pip install .[macos]

mogrify -format jpg -path /Users/akhileshsharma/Documents/Lumina/chunk-my-docs/services/task/input/table_ocr/jpg /Users/akhileshsharma/Documents/Lumina/chunk-my-docs/services/task/input/table_ocr/*.png
mogrify -format jpg -path /Users/akhileshsharma/Documents/Lumina/chunk-my-docs/services/task/input/latex_ocr/jpg /Users/akhileshsharma/Documents/Lumina/chunk-my-docs/services/task/input/latex_ocr/*.png


curl -X POST localhost:3000/table_to_html \
    -F 'file=@/Users/akhileshsharma/Documents/Lumina/chunk-my-docs/services/task/input/llm/c86ce3d2-295a-4d1b-a8c5-6fe427b6da21.jpg' \
    -o output/output.html

curl -X POST localhost:3000/table_to_html \
    -F 'file=@/Users/akhileshsharma/Downloads/Screenshot 2024-09-30 at 5.54.49 PM.png' \
    -o output/output.html

curl -X POST localhost:3000/table_to_html \
    -F 'file=@/Users/akhileshsharma/Downloads/ebdc985f-5e5f-401d-bd47-b11487047a7a.jpg' \
    -o output/output.html