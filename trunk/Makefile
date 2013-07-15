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
	mv book.pdf thinkbayes.pdf
	evince thinkbayes.pdf

hevea:
	sed 's/\(figs\/[^.]*\).\(pdf\|png\)/\1.eps/' book.tex > thinkbayes.tex
	rm -rf html
	mkdir html
	hevea -O -e latexonly htmlonly thinkbayes
# the following greps are a kludge to prevent imagen from seeing
# the definitions in latexonly, and to avoid headers on the images
	grep -v latexonly thinkbayes.image.tex > a; mv a thinkbayes.image.tex
	grep -v fancyhdr thinkbayes.image.tex > a; mv a thinkbayes.image.tex
	imagen -png thinkbayes
	hacha thinkbayes.html
	cp up.png next.png back.png html
	mv index.html thinkbayes.css thinkbayes*.html  html
	mv thinkbayes*.png *motif.gif html

DEST = /home/downey/public_html/greent/thinkbayes

distrib:
	rm -rf dist
	mkdir dist dist/tex dist/tex/figs
	rsync -a thinkbayes.pdf html cover_nolines.png dist
	rsync -a Makefile book.tex latexonly htmlonly dist/tex
	# rsync -a figs/*.fig figs/*.eps dist/tex/figs
	cd dist; zip -r thinkbayes.tex.zip tex
	cd dist; zip -r thinkbayes.html.zip html
	rsync -a dist/* $(DEST)
	chmod -R o+r $(DEST)/*
	cd /home/downey/public_html/greent; sh back

plastex:
	# Before running plastex, we need the current directory in PYTHONPATH
	# export PYTHONPATH=$PYTHONPATH:.
	rm -rf /home/downey/thinkbayes/trunk/book/.svn
	plastex --renderer=DocBook --theme=book --image-resolution=300 --filename=book.xml book.tex
	rm -rf /home/downey/thinkbayes/trunk/book/.svn

xxe:
	xmlcopyeditor ~/thinkstats/trunk/thinkbayes/book/book.xml &
	#~/Downloads/xxe-perso-4_8_0/bin/xxe book/book.xml

lint:
	xmllint -noout book/book.xml

OREILLY = ~/oreilly/thinkbayes

oreilly:
	rsync -a book/ $(OREILLY)
	rsync -a figs/* $(OREILLY)/figs/
	cd $(OREILLY); git add .
	cd $(OREILLY); git commit -m "Automated check in."
	cd $(OREILLY); git push origin master
	#cp thinkbayes.pdf ~/oreilly/1234000001745/pdf

clean:
	rm -f *~ *.aux *.log *.dvi *.idx *.ilg *.ind *.toc



