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

DEST = /home/downey/web/greent/thinkdsp

distrib:
	rm -rf dist
	mkdir dist
	rsync -a thinkdsp.pdf html dist
	rsync -a dist/* $(DEST)
	chmod -R o+r $(DEST)/*
	cd $(DEST)/..; sh back

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
