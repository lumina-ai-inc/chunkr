from markdownify import markdownify as md

if __name__ == "__main__":
    html = "<html><body><table><tbody><tr><td>Data Dist.</td><td>Method</td><td>MNLI (matched)</td><td>MNLI (mismatched)</td><td>SST2</td><td>QQP</td><td>QNLI</td></tr><tr><td rowspan=\"2\">i.i.d.</td><td>LoRA</td><td>86.90</td><td>87.15</td><td>94.42</td><td>84.47</td><td>91.38</td></tr><tr><td>FFA-LoRA</td><td>87.13</td><td>87.21</td><td>95.14</td><td>86.31</td><td>92.64</td></tr><tr><td rowspan=\"2\">mild het.</td><td>LoRA</td><td>87.01</td><td>87.33</td><td>93.55</td><td>84.41</td><td>91.36</td></tr><tr><td>FFA-LoRA</td><td>87.04</td><td>87.36</td><td>94.10</td><td>85.33</td><td>91.62</td></tr><tr><td rowspan=\"2\">severe het.</td><td>LoRA</td><td>82.03</td><td>82.50</td><td>94.32</td><td>83.51</td><td>88.95</td></tr><tr><td>FFA-LoRA</td><td>85.05</td><td>85.62</td><td>94.32</td><td>84.35</td><td>90.35</td></tr></tbody></table></body></html>"
    print(md(html))
