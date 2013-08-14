FILES = thinkdsp.py violin.py

DATA = 92002__jcveliz__violin-origional.wav

DOCS = thinkdsp.html

DOCPY = thinkdsp.py

DEST = /home/downey/public_html/greent/thinkdsp

FIGS = sinusoid1.pdf sinusoid1.eps \
violin1.pdf  violin1.eps  \
tuning1.pdf  tuning1.eps  \
violin2.pdf  violin2.eps  \
example1.pdf example1.eps \

all_figs:
	python violin.py

FIGDEST = ../trunk/figs

%.html: %.py
	pydoc -w $<

code:
	zip -r thinkstats.code.zip $(FILES)
	rsync -a thinkstats.code.zip $(FILES) $(DATA) $(DEST)
	rsync -a $(DOCS) $(DEST)
	chmod -R o+r $(DEST)/*

figs:
	rsync -a $(FIGS) $(FIGDEST)

.PHONY: docs $(DOCPY)

docs: $(DOCPY)

$(DOCPY):
	pydoc -w ./$@

