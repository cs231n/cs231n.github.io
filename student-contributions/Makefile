# You want latexmk to *always* run, because make does not have all the info.
# Also, include non-file targets in .PHONY so they are run regardless of any
# file of the given name existing.
.PHONY: BackPropagationBasicMatrixOperations.pdf

# The first rule in a Makefile is the one executed by default ("make"). It
# should always be the "all" rule, so that "make" and "make all" are identical.
all: BackPropagationBasicMatrixOperations.pdf

# MAIN LATEXMK RULE

CC = latexmk 

# -pdf tells latexmk to generate PDF directly (instead of DVI).
# -pdflatex="" tells latexmk to call a specific backend with specific options.
# -use-make tells latexmk to call make for generating missing files.

# -interaction=nonstopmode keeps the pdflatex backend from stopping at a
# missing file reference and interactively asking you for an alternative.
CFLAGS = -pdf -pdflatex="pdflatex -interaction=nonstopmode" -use-make

BackPropagationBasicMatrixOperations.pdf: BackPropagationBasicMatrixOperations.tex
	$(CC) $(CFLAGS) BackPropagationBasicMatrixOperations.tex

# latexmk
# -CA clean up (remove) all nonessential files.
clean:
	$(CC) -CA
