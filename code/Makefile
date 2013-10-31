FILES = thinkdsp.py violin.py example1.py example2.py example3.py \
thinkplot.py rednoise.py noise.py

DATA = 92002__jcveliz__violin-origional.wav \
18871__zippi1__sound-bell-440hz.wav

DOCS = thinkdsp.html thinkplot.html

DOCPY = thinkdsp.py thinkplot.py

DEST = /home/downey/public_html/greent/thinkdsp

FIGS = sinusoid1.pdf sinusoid1.eps \
violin1.pdf  violin1.eps  \
tuning1.pdf  tuning1.eps  \
violin2.pdf  violin2.eps  \
example1.pdf example1.eps \
aliasing-3.eps  square-100-1.eps  triangle-1100-1.eps  triangle-200-1.eps \
aliasing-3.pdf  square-100-1.pdf  triangle-1100-1.pdf  triangle-200-1.pdf \
square-100-2.eps  triangle-1100-2.eps  triangle-200-2.eps \
square-100-2.pdf  triangle-1100-2.pdf  triangle-200-2.pdf \
chirp1.pdf chirp1.eps \
chirp2.pdf chirp2.eps \
windowing1.pdf windowing1.eps \
windowing2.pdf windowing2.eps \
windowing3.pdf windowing3.eps \
whitenoise0.pdf whitenoise0.eps \
whitenoise1.pdf whitenoise1.eps \
whitenoise2.pdf whitenoise2.eps \
whitenoise3.pdf whitenoise3.eps \
whitenoise4.pdf whitenoise4.eps \
whitenoise5.pdf whitenoise5.eps \
rednoise0.pdf rednoise0.eps \
rednoise1.pdf rednoise1.eps \
rednoise2.pdf rednoise2.eps \
rednoise3.pdf rednoise3.eps \
rednoise4.pdf rednoise4.eps \
rednoise5.pdf rednoise5.eps \



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
	cd /home/downey/public_html/greent; sh back

figs:
	rsync -a $(FIGS) $(FIGDEST)

.PHONY: docs $(DOCPY)

docs: $(DOCPY)

$(DOCPY):
	pydoc -w ./$@

