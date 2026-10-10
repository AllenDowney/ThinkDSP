# instructions
#
# GitHub book/ is canonical for the 1e PDF and EPUB (PROJECT_BOARD Task 10/11).
# GTP holds stable URLs; Hevea HTML is frozen. ThinkDSP2 is the HTML to read.
#
# make            # PDF, if the print book changed
# make epub
# make distrib           # copy PDF + EPUB into book/ for commit and push
# make publish-gtp-dry   # rsync -n PDF+EPUB to GTP (review this first)
# make publish-gtp       # actually rsync PDF+EPUB (never --delete, never html/)

PDFFLAGS = -dCompatibilityLevel=1.4 -dPDFSETTINGS=/prepress \
           -dCompressPages=true -dUseFlateCompression=true  \
           -dEmbedAllFonts=true -dSubsetFonts=true -dMaxSubsetPct=100


all:	book.tex
	pdflatex book
	makeindex book.idx
	pdflatex book
	mv book.pdf thinkdsp.pdf
	evince thinkdsp.pdf

HEVEABUILDLOG=build.log

hevea:
	sed 's/\(figs\/[^.]*\).\(pdf\|png\)/\1.eps/' book.tex > thinkdsp.tex
	rm -rf html
	mkdir html
	hevea macros.hva -O -e latexonly htmlonly thinkdsp 2>&1 | tee ${HEVEABUILDLOG}
# the following greps are a kludge to prevent imagen from seeing
# the definitions in latexonly, and to avoid headers on the images
	grep -v latexonly thinkdsp.image.tex > a; mv a thinkdsp.image.tex
	sed s/\\\\usepackage{fancyhdr}// < thinkdsp.image.tex > a; mv a thinkdsp.image.tex
	imagen -png thinkdsp
	hacha thinkdsp.html
	cp up.png next.png back.png html
	cp custom.css thinkdsp.css
	mv index.html thinkdsp.css thinkdsp*.html thinkdsp*.png *motif.gif html

# ---------------------------------------------------------------------------
# EPUB
#
# Builds a reflowable EPUB3 from the same book.tex used for the PDF, without
# modifying it.  Requires pandoc (>= 3) and poppler (for pdftocairo); see the
# "Building the book" section of the top-level README.
#
#	make epub		build build/epub/thinkdsp.epub
#	make epub-check		validate it with epubcheck
#	make epub-clean		remove build/epub
#
# Override the figure resolution with e.g. `make epub EPUB_DPI=200`.
# ---------------------------------------------------------------------------

EPUB_DPI ?= 150
EPUB_DIR  = build/epub
EPUB_TEX  = $(EPUB_DIR)/book.tex
EPUB_OUT  = $(EPUB_DIR)/thinkdsp.epub

# Read the version out of book.tex so it is not duplicated here.  The title,
# subtitle and author reach the EPUB from book.tex too: pandoc picks up \title
# and \author, and the retained htmlonly block supplies the visible title page.
EPUB_VERSION = $(shell sed -n 's/.*newcommand{\\theversion}{\([^}]*\)}.*/\1/p' book.tex)

# Only the figures book.tex actually includes; figs/ holds ~40 more that the
# first edition no longer references.
EPUB_FIG_NAMES = $(shell sed -n 's|.*{figs/\([^}]*\)\.pdf}.*|\1|p' book.tex | sort -u)
EPUB_FIGS = $(patsubst %,$(EPUB_DIR)/figs/%.png,$(EPUB_FIG_NAMES))

.PHONY: epub epub-check epub-clean distrib publish-gtp-dry publish-gtp

epub: $(EPUB_OUT)

# Figures are vector PDFs, which are not a valid EPUB image type.  At 150 dpi
# a default 8x6in matplotlib figure becomes 1200px wide, which matches a
# typical e-ink panel without wasting bytes.
$(EPUB_DIR)/figs/%.png: figs/%.pdf
	@mkdir -p $(@D)
	pdftocairo -png -r $(EPUB_DPI) -singlefile $< $(basename $@)

# Four substitutions are all that separate the print source from EPUB input:
#   1. \input{latexonly} pulls in print-only page geometry and fancyhdr setup.
#   2. The latexonly blocks are the print title and copyright pages.  The
#      htmlonly block is kept: it is the author's own title page and CC
#      license notice, which is exactly the front matter an EPUB wants.
#   3/4. Point figure includes at the PNGs and drop the print sizes, so the
#      stylesheet can scale them to the reader's screen instead.
$(EPUB_TEX): book.tex
	@mkdir -p $(@D)
	sed -e '/^\\input{latexonly}$$/d' \
	    -e '/^\\begin{latexonly}$$/,/^\\end{latexonly}$$/d' \
	    -e 's|{figs/\([^}]*\)\.pdf}|{figs/\1.png}|g' \
	    -e 's|\\includegraphics\[[^]]*\]|\\includegraphics|g' \
	    $< > $@

# --resource-path lets the rewritten figs/*.png references resolve against the
# build directory.  Pandoc stamps the current time into dcterms:modified unless
# SOURCE_DATE_EPOCH is set; pinning it to 2014-01-01 (the book's copyright year)
# makes repeated builds of an unchanged source produce a byte-identical file.
$(EPUB_OUT): $(EPUB_TEX) $(EPUB_FIGS) epub/epub.css
	SOURCE_DATE_EPOCH=$${SOURCE_DATE_EPOCH:-1388534400} pandoc $(EPUB_TEX) \
	    --from=latex \
	    --to=epub3 \
	    --standalone \
	    --toc --toc-depth=2 \
	    --mathml \
	    --css=epub/epub.css \
	    --resource-path=$(EPUB_DIR) \
	    --epub-title-page=false \
	    --metadata lang=en-US \
	    --metadata date=2014 \
	    --metadata identifier="http://think-dsp.com/thinkdsp-$(EPUB_VERSION)" \
	    --metadata rights="Copyright 2014 Allen B. Downey. Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (https://creativecommons.org/licenses/by-nc-sa/4.0/)." \
	    --output=$@
	@echo ">>> $(EPUB_OUT)"

epub-check: $(EPUB_OUT)
	epubcheck $(EPUB_OUT)

epub-clean:
	rm -rf $(EPUB_DIR)


# Stage PDF and EPUB next to the LaTeX sources for commit + push to GitHub.
# README download links point at raw GitHub URLs under book/.
distrib: $(EPUB_OUT)
	@test -f thinkdsp.pdf || (echo ">>> missing thinkdsp.pdf; run make first" && exit 1)
	cp -f $(EPUB_OUT) thinkdsp.epub
	@echo ">>> Staged for GitHub (commit and push):"
	@ls -lh thinkdsp.pdf thinkdsp.epub

# Mirror PDF + EPUB to Green Tea Press. Requires ~/.ssh/config Host gtp
# (HostName greenteapress.com, User adowney). Two files only; no --delete;
# never html/. Dry-run first: make publish-gtp-dry
GTP_HOST ?= gtp
GTP_DIR  ?= greenteapress.com/thinkdsp
RSYNC_GTP = rsync -avz --itemize-changes thinkdsp.pdf thinkdsp.epub \
	$(GTP_HOST):$(GTP_DIR)/

publish-gtp-dry:
	@test -f thinkdsp.pdf || (echo ">>> missing thinkdsp.pdf; run make first" && exit 1)
	@test -f thinkdsp.epub || (echo ">>> missing thinkdsp.epub; run make distrib" && exit 1)
	$(RSYNC_GTP) -n
	@echo ">>> Dry run only. If that looks right: make publish-gtp"

publish-gtp:
	@test -f thinkdsp.pdf || (echo ">>> missing thinkdsp.pdf; run make first" && exit 1)
	@test -f thinkdsp.epub || (echo ">>> missing thinkdsp.epub; run make distrib" && exit 1)
	$(RSYNC_GTP)
	@echo ">>> Published to $(GTP_HOST):$(GTP_DIR)/"

plastex:
	# Before running plastex, we need the current directory in PYTHONPATH
	# export PYTHONPATH=$PYTHONPATH:.
	rm -rf /home/downey/thinkdsp/trunk/book/.svn
	plastex --renderer=DocBook --theme=book --image-resolution=300 --filename=book.xml book.tex
	rm -rf /home/downey/thinkdsp/trunk/book/.svn

xxe:
	xmlcopyeditor ~/ThinkDSP/book/book/book.xml &

lint:
	xmllint -noout book/book.xml

OREILLY = atlas

oreilly:
	rsync -a book/ $(OREILLY)
	rsync -a figs/* $(OREILLY)/figs/
	cd $(OREILLY); git add .
	cd $(OREILLY); git commit -m "Automated check in."
	cd $(OREILLY); git push

clean:
	rm -f *~ *.aux *.log *.dvi *.idx *.ilg *.ind *.toc


pdfpextr = \
    gs -sDEVICE=pdfwrite -dNOPAUSE -dBATCH -dSAFER \
       -dFirstPage=$(1) \
       -dLastPage=$(2) \
       -sOutputFile=$(4) \
       $(3)

split:
	$(call pdfpextr,5,7,thinkos.pdf,chap00.pdf)
	$(call pdfpextr,13,20,thinkos.pdf,chap01.pdf)
	$(call pdfpextr,21,26,thinkos.pdf,chap02.pdf)
	$(call pdfpextr,27,34,thinkos.pdf,chap03.pdf)
	$(call pdfpextr,35,42,thinkos.pdf,chap04.pdf)
	$(call pdfpextr,43,50,thinkos.pdf,chap05.pdf)
	$(call pdfpextr,51,56,thinkos.pdf,chap06.pdf)
	$(call pdfpextr,57,68,thinkos.pdf,chap07.pdf)
	$(call pdfpextr,69,74,thinkos.pdf,chap08.pdf)
	$(call pdfpextr,75,82,thinkos.pdf,chap09.pdf)
	$(call pdfpextr,83,94,thinkos.pdf,chap10.pdf)
	$(call pdfpextr,95,101,thinkos.pdf,chap11.pdf)
