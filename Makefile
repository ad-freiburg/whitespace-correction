.PHONY: checkstyle
checkstyle:
	@echo "Running flake8\n-----------------"
	flake8 src/whitespace_correction

.PHONY: tests
tests:
	pytest tests -n auto --disable-pytest-warnings

.PHONY: pypi_release
pypi_release:
	rm -r dist
	python -m build
	python -m twine upload dist/*
