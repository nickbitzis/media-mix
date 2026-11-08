PORT ?= 8859

.PHONY: install test lint format run eval gate screenshot docker-build clean

install:
	uv sync --python 3.12

test:
	uv run --python 3.12 --group dev pytest -q

lint:
	uvx ruff check .

format:
	uvx ruff format .
	uvx ruff check . --fix

run:  ## fits the study at startup (about 10 seconds), then serves the API and UI
	PORT=$(PORT) uv run mediamix serve

eval:  ## full study, writes reports/eval.json and reports/summary.md
	uv run mediamix eval

gate:  ## exit 1 unless influencer, email, and social + display combined recover true ROI within 30%
	uv run mediamix eval --gate --out reports-ci

screenshot:  ## needs the server running on $(PORT) and Google Chrome installed (macOS path)
	"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --hide-scrollbars \
		--virtual-time-budget=10000 --window-size=1440,3000 --screenshot=docs/screenshot.png \
		"http://127.0.0.1:$(PORT)/"

docker-build:
	docker build -t media-mix .

clean:
	rm -rf .pytest_cache .ruff_cache reports-ci
