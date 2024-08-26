import json

def convert_to_markdown(input_file, output_file):
    with open(input_file, 'r') as f:
        data = json.load(f)

    markdown = ""
    for item in data:
        text = item['text'].strip()
        if not text:
            continue

        if item['type'] == 'Section header':
            markdown += f"## {text}\n\n"
        elif item['type'] == 'Text':
            markdown += f"{text}\n\n"
        elif item['type'] == 'List item':
            markdown += f"- {text}\n"
        elif item['type'] == 'Table':
            markdown += f"```\n{text}\n```\n\n"
        elif item['type'] == 'Caption':
            markdown += f"*{text}*\n\n"
        elif item['type'] == 'Page footer':
            markdown += f"---\n{text}\n---\n\n"
        else:
            markdown += f"{text}\n\n"

    with open(output_file, 'w') as f:
        f.write(markdown)

# Usage
convert_to_markdown('output/00c08086-9837-5551-8133-4e22ac28c6a5-HighQuality/bounding_boxes.json', 'output.md')
