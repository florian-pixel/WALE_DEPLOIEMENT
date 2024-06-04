.DEFAULT_GOAL := dev

# Par défaut, Stackhero pour Python exécutera la règle "run". Nous demandons à Makefile d'exécuter la règle `prod` dans ce cas.
run: prod

prod:
	ENV=production gunicorn app:app \
		--error-logfile - \
		-b 0.0.0.0:8080

dev:
	python app.py
