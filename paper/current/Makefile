MAIN := main
TEX := latexmk 
MON_OPT := -pvc
CLN_OPT := -CA

.PHONY: all main-monitor supp-monitor wordcount clean

all: ${MAIN}.pdf 

${MAIN}.pdf:
	${TEX} ${MAIN}.tex

main-monitor:
	${TEX} ${MON_OPT} ${MAIN}.tex

clean:
	${TEX} ${CLN_OPT} ${MAIN}.tex

wordcount:
	detex -l -n -w ${MAIN}.tex | wc -w

