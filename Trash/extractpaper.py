from pdfminer.high_level import extract_text

# Path to your PDF file
pdf_path = 'paper.pdf'

# Extract text from the PDF
text = extract_text(pdf_path)

# Save the extracted text to a file
with open('output.txt', 'w') as f:
    f.write(text)

print("Text extraction completed!")
