.PHONY: checkstyle
checkstyle:
	@echo "Running flake8\n-----------------"
	flake8 src/trt

.PHONY: tests
tests:
	pytest tests -n auto --disable-pytest-warnings
