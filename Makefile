.PHONY: checkstyle
checkstyle:
	@echo "Running flake8\n-----------------"
	flake8 src/whitespace_repair

.PHONY: tests
tests:
	pytest tests -n auto --disable-pytest-warnings
