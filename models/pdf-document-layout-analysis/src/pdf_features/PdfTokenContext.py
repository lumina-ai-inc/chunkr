from dataclasses import dataclass


@dataclass
class PdfTokenContext:
    right_of_token_on_the_left = 0
    left_of_token_on_the_left = 0
    left_of_token_on_the_right = 0
    right_of_token_on_the_right = 0
