FILES = thinkdsp.py violin.py example1.py example2.py example3.py \
thinkplot.py rednoise.py noise.py

DATA = 92002__jcveliz__violin-origional.wav \
18871__zippi1__sound-bell-440hz.wav

DOCS = thinkdsp.html thinkplot.html

DOCPY = thinkdsp.py thinkplot.py

DEST = /home/downey/public_html/greent/thinkdsp

FIGS = \
sounds1.pdf sounds1.eps  \
sounds2.pdf sounds2.eps  \
sounds3.pdf sounds3.eps  \
sounds4.pdf sounds4.eps  \
aliasing1.eps  square-100-1.eps  triangle-1100-1.eps  triangle-200-1.eps \
aliasing1.pdf  square-100-1.pdf  triangle-1100-1.pdf  triangle-200-1.pdf \
square-100-2.eps  triangle-1100-2.eps  triangle-200-2.eps \
square-100-2.pdf  triangle-1100-2.pdf  triangle-200-2.pdf \
chirp1.pdf chirp1.eps \
chirp2.pdf chirp2.eps \
chirp3.pdf chirp3.eps \
windowing1.pdf windowing1.eps \
windowing2.pdf windowing2.eps \
windowing3.pdf windowing3.eps \
whitenoise0.pdf whitenoise0.eps \
whitenoise1.pdf whitenoise1.eps \
whitenoise2.pdf whitenoise2.eps \
rednoise0.pdf rednoise0.eps \
rednoise3.pdf rednoise3.eps \
pinknoise0.pdf pinknoise0.eps \
noise-triple.pdf noise-triple.eps \
noise1.pdf noise1.eps \
autocorr1.pdf autocorr1.eps \
autocorr2.pdf autocorr2.eps \
autocorr3.pdf autocorr3.eps \
autocorr4.pdf autocorr4.eps \
autocorr5.pdf autocorr5.eps \
autocorr6.pdf autocorr6.eps \
autocorr7.pdf autocorr7.eps \
autocorr8.pdf autocorr8.eps \
autocorr9.pdf autocorr9.eps \
dft1.pdf dft1.eps \
dft2.pdf dft2.eps \
dft3.pdf dft3.eps \
convolution1.pdf convolution1.eps \
convolution2.pdf convolution2.eps \
convolution4.pdf convolution4.eps \
convolution5.pdf convolution5.eps \
convolution6.pdf convolution6.eps \
convolution7.pdf convolution7.eps \
convolution8.pdf convolution8.eps \
convolution9.pdf convolution9.eps \
systems1.pdf systems1.eps \
systems2.pdf systems2.eps \
systems3.pdf systems3.eps \
systems4.pdf systems4.eps \
systems5.pdf systems5.eps \
systems6.pdf systems6.eps \
systems7.pdf systems7.eps \
systems8.pdf systems8.eps \
sampling1.pdf sampling1.eps \
sampling2.pdf sampling2.eps \
sampling3.pdf sampling3.eps \
sampling4.pdf sampling4.eps \
sampling5.pdf sampling5.eps \
sampling6.pdf sampling6.eps \
sampling7.pdf sampling7.eps \
sampling8.pdf sampling8.eps \

all_figs:
	python violin.py
	python sampling.py

FIGDEST = ../book/figs

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
	rsync -a $(DOCS) $(DEST)
	cd /home/downey/public_html/greent; sh back

$(DOCPY):
	pydoc -w ./$@

