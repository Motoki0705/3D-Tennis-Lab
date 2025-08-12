import jupytext
import sys

# Get paths from command line arguments
md_file = sys.argv[1]
ipynb_file = sys.argv[2]

# Read markdown file
notebook = jupytext.read(md_file)

# Write to notebook file
jupytext.write(notebook, ipynb_file)

print(f"Converted {md_file} to {ipynb_file}.")
