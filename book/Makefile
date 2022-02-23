# instructions

# Currently the staging area for greenteapress.com is on bob at
# /home/downey/web/greenteapress

# To build and sync a new release:
# make
# make hevea
# make distrib

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

DEST = /home/downey/web/greenteapress/thinkdsp

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
