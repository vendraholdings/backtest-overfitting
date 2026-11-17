UV ?= uv
RUN = $(UV) run

.PHONY: all test results figures hero paper readme site serve clean

all: test results figures hero paper readme site

test:
	$(RUN) pytest -q

results:
	$(RUN) python scripts/run_all.py

figures:
	$(RUN) python scripts/make_figures.py

hero:
	$(RUN) python scripts/make_hero.py

paper:
	$(RUN) python scripts/build_paper.py

readme:
	$(RUN) python scripts/build_readme.py

site:
	$(RUN) python site/build.py

serve:
	$(RUN) python -m http.server 8000 --directory site/dist

clean:
	rm -rf site/dist results paper/paper.typ paper/paper.pdf paper/figures docs
