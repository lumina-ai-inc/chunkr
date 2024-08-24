def create_html_table(data):
    html = "<table border='1'>"
    for row_data in data.values():
        html += "<tr>"
        for cell in row_data:
            if cell == "":
                html += "<td></td>"
            else:
                html += f"<td>{cell}</td>"
        html += "</tr>"
    html += "</table>"
    return html