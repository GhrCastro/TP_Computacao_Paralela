# Baseline sequencial — Felipe

## Dados reais
```sh
mkdir -p data bin
curl -fL https://kdd.ics.uci.edu/databases/covertype/covtype.data.gz -o data/covtype.data.gz
gzip -dk data/covtype.data.gz
wc -l data/covtype.data
```
Esperado: 581012 linhas, 54 atributos utilizados. Fonte: UCI Covertype.
Nao versionar os dados nem executaveis.

## Compilacao e testes
```sh
gcc -O3 -std=c11 -Wall -Wextra -Iinclude src/main.c src/common.c src/seq.c -lm -o bin/kmeans_seq
gcc -O3 -std=c11 -Wall -Wextra -Iinclude tests/test_seq.c src/common.c src/seq.c -lm -o bin/test_seq
./bin/test_seq
```
O Makefile integrado pertence a Gustavo; estes comandos permitem testar antes da integracao.

## Calibracao
```sh
lscpu
uname -a
gcc --version
/usr/bin/time -p ./bin/kmeans_seq data/covtype.data 64 50 42
```
Registrar n=581012, d=54, inertia finita, time_s e real/user/sys.
Se time_s for inferior a 10 s, aumentar iters (por exemplo, 100) e repetir.
Comunicar os parametros efetivamente medidos ao grupo para todas as versoes.
Atualizar o comentario de src/seq.c com hardware, flags e tempo real.
Status: medicao no equipamento final pendente; nenhum tempo estimado apresentado como medido.
