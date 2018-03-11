LATEX = latex

DVIPS = dvips

PDFFLAGS = -dCompatibilityLevel=1.4 -dPDFSETTINGS=/prepress \
           -dCompressPages=true -dUseFlateCompression=true  \
           -dEmbedAllFonts=true -dSubsetFonts=true -dMaxSubsetPct=100


%.dvi: %.tex
	$(LATEX) $<

%.ps: %.dvi
	$(DVIPS) -o $@ $<

%.pdf: %.ps
	ps2pdf $(PDFFLAGS) $<

all:	book.tex
	pdflatex book
	makeindex book.idx
	mv book.pdf thinkdsp.pdf
	evince thinkdsp.pdf

html: thinkdsp.html

book-svg.tex: book.tex
	sed 's/figs\/\([^.]*\).pdf/figs\/\1.svg/' book.tex | grep -v '\\input{section.sty}' > book-svg.tex

thinkdsp.html: book-svg.tex
	pandoc --standalone --mathjax --from=latex --to=html5 $< --output $@

epub: thinkdsp.epub
	
thinkdsp.epub: book-svg.tex
	pandoc book-svg.tex --output thinkdsp.epub

DEST = /home/downey/public_html/greent/thinkdsp

distrib:
	rm -rf dist
	mkdir dist dist/tex dist/tex/figs
	rsync -a thinkdsp.pdf html dist
	rsync -a Makefile book.tex latexonly htmlonly dist/tex
	# rsync -a figs/*.fig figs/*.eps dist/tex/figs
	cd dist; zip -r thinkdsp.tex.zip tex
	cd dist; zip -r thinkdsp.html.zip html
	rsync -a dist/* $(DEST)
	chmod -R o+r $(DEST)/*
	cd /home/downey/public_html/greent; sh back

plastex:
	# Before running plastex, we need the current directory in PYTHONPATH
	# export PYTHONPATH=$PYTHONPATH:.
	rm -rf /home/downey/thinkdsp/trunk/book/.svn
	plastex --renderer=DocBook --theme=book --image-resolution=300 --filename=book.xml book.tex
	rm -rf /home/downey/thinkdsp/trunk/book/.svn

xxe:
	xmlcopyeditor ~/ThinkDSP/book/book/book.xml &
	#~/Downloads/xxe-perso-4_8_0/bin/xxe book/book.xml

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



