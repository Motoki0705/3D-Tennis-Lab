import jupytext
import sys

# Get paths from command line arguments
ipynb_file = sys.argv[1]
md_file = sys.argv[2]

# Read notebook file
notebook = jupytext.read(ipynb_file)

# Write to markdown file
jupytext.write(notebook, md_file)

print(f"Converted {ipynb_file} to {md_file}.")
