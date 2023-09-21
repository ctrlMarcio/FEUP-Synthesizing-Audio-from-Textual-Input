# Define the name of the Beamer project
PROJECT_NAME = presentation

# Define the name of the main TeX file
MAIN_TEX_FILE = $(PROJECT_NAME).tex

# Define the name of the output PDF file
OUTPUT_PDF_FILE = $(PROJECT_NAME).pdf

# Define the TeX compiler to use
TEX_COMPILER = pdflatex

# Define any additional flags to pass to the TeX compiler
TEX_FLAGS = -shell-escape

# Define the command to clean up the build directory
CLEAN_COMMAND = rm -f *.aux *.log *.nav *.out *.snm *.toc *.bcf *.xml

# Define the default target
all: $(OUTPUT_PDF_FILE)

# Define the rule to build the PDF file
$(OUTPUT_PDF_FILE): $(MAIN_TEX_FILE)
	$(TEX_COMPILER) $(TEX_FLAGS) $(MAIN_TEX_FILE)

# Define the rule to clean up the build directory
clean:
	$(CLEAN_COMMAND)