import pdf2image
import argparse
import logging
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--pdf",
                        required=True,
                        help="Path to the PDF file")
    parser.add_argument("--output",
                        required=False,
                        default="images",
                        help="Path to the output folder")
    parser.add_argument("--format",
                        required=False,
                        default="png",
                        help="Output image format")

    args = parser.parse_args()

    # Create the output folder if it doesn't exist
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # Convert the PDF to images
    images = pdf2image.convert_from_path(
        args.pdf,
        dpi=72,  # standard dpi used by pdfplumber is 72
        fmt=args.format)

    # Save the images
    for i, image in enumerate(images):
        image.save(
            os.path.join(
                args.output, f"page_{i}.png"))

    logging.info(f"PDF converted to images and saved at {args.output}")
