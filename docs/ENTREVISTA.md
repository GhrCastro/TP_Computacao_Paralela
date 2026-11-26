# Guia para a entrevista — K-means sequencial, OpenMP e CUDA

Este guia descreve a implementacao entregue. Os pontos centrais aparecem em
[`src/seq.c`](../src/seq.c), [`src/omp.c`](../src/omp.c),
[`src/cuda.cu`](../src/cuda.cu) e [`src/common.c`](../src/common.c).

## Narrativa de dois minutos

K-means e aprendizado nao supervisionado: particiona N pontos em K clusters sem
receber rotulos. Lloyd alterna uma fase MAP, que associa cada ponto ao centroide
mais proximo, e uma fase REDUCE, que soma os pontos por cluster; a media vira o
novo centroide. Repetimos I vezes. O custo dominante e O(I*N*K*D), porque cada
ponto compara D coordenadas com K centroides.

Na CPU sequencial ha um laco sobre pontos. OpenMP divide esse laco com
`schedule(static)` e faz reducao de arrays privados, evitando corrida. CUDA usa
uma thread por ponto, acumulacao em shared memory por bloco, parciais globais e
kernels de reducao/atualizacao. Iteracoes nao sao paralelas porque a seguinte
depende dos centroides anteriores. No Covertype real, a inercia foi igual nas
tres versoes. OpenMP chegou a 5,84x no i5-12400 e CUDA a 11,26x contra o
sequencial medido na mesma maquina da GTX 1660 Ti.

## Conceitos que precisam estar claros

- **Centroide:** media vetorial dos membros de um cluster; nao precisa ser um
  registro real.
- **Distancia quadratica:** soma de `(p[j]-c[j])^2`. O `sqrt` e monotono e pode
  ser omitido na escolha do menor.
- **Inercia/SSE:** soma da menor distancia quadratica de cada ponto aos
  centroides finais. Menor nao significa automaticamente melhor K.
- **Complexidade:** O(I*N*K*D); a atualizacao O(K*D) e pequena diante da busca.
- **Paralelismo de dados:** pontos sao independentes enquanto os centroides de
  uma iteracao permanecem somente leitura.
- **Dependencia:** a iteracao t+1 precisa dos centroides calculados em t.
- **Condicao de corrida:** duas threads acessam a mesma posicao, ao menos uma
  escreve, sem sincronizacao. Labels nao correm porque cada ponto tem um dono;
  somas/contagens precisam de reduction ou atomicAdd.
- **OpenMP reduction:** cada thread recebe acumuladores privados; o runtime os
  combina na barreira. Evita uma operacao atomica por coordenada.
- **Threads OpenMP efetivas:** `OMP_NUM_THREADS` solicita uma equipe e
  `omp_get_max_threads()` fora da regiao indica apenas um limite. O programa
  registra `omp_get_num_threads()` dentro da regiao `parallel` que contem o
  `for`, comprovando a equipe que realmente executou o K-means.
- **schedule(static):** reparte intervalos antes do laco, com overhead baixo. O
  trabalho por ponto e regular: todos comparam K*D coordenadas.
- **Fork-Join:** threads OpenMP se separam no `parallel for` e se juntam na
  barreira implicita antes de atualizar centroides.
- **Grid/bloco/thread:** grid e o conjunto de blocos; bloco agrupa threads que
  cooperam e compartilham shared memory; thread processa um indice de ponto.
- **Warp:** grupo de 32 threads executado em SIMT. Divergencias dentro do warp
  serializam caminhos, embora este kernel tenha controle bastante uniforme.
- **Registradores:** memoria privada e muito rapida da thread; variaveis como
  distancia e melhor label tendem a ficar neles.
- **Shared memory:** pequena, por bloco, programavel e rapida; guarda K*D somas
  e K contagens parciais.
- **Memoria global:** grande e de maior latencia; guarda dataset, centroides,
  labels e vetores parciais.
- **atomicAdd:** faz leitura-modificacao-escrita indivisivel. Garante soma
  correta, mas acessos ao mesmo endereco serializam. Por isso as atomicas foram
  confinadas a cada bloco.
- **HtoD/DtoH:** dataset e centroides iniciais vao do host ao device uma vez;
  centroides e labels retornam ao final. Nao ha copia do dataset por iteracao.
- **Sincronizacao:** `__syncthreads()` protege fases dentro de um bloco; kernels
  no mesmo stream executam em ordem; `cudaDeviceSynchronize()` garante conclusao.
- **Speedup:** T_sequencial/T_paralelo. **Eficiencia:** speedup/threads. GPU nao
  usa a mesma metrica simples de eficiencia por thread.
- **Amdahl:** uma fracao sequencial limita o speedup, mesmo com muitos recursos.
- **Oversubscription:** mais threads OpenMP que CPUs logicas; aumenta disputa e
  trocas sem criar capacidade fisica.
- **K-means nao e rede neural:** nao ha neuronios, pesos treinados por gradiente,
  backpropagation nem rotulos. Ha alternancia entre atribuicao e medias.

## Caminho do codigo

1. `main.c` valida CLI, chama `load_dataset`, `init_centroids` e inicia o relogio.
2. Uma unica implementacao de `kmeans_run` e vinculada ao binario.
3. `seq.c` e a referencia; `omp.c` paraleliza pontos; `cuda.cu` gerencia device.
4. O relogio para depois de `kmeans_run`; `calculate_inertia` fica fora.
5. `common.c` libera tudo e a CLI imprime os campos reproduziveis.

## Perguntas provaveis e respostas

1. **Por que K-means e nao supervisionado?**
   Porque os rotulos de classe sao descartados. O algoritmo descobre grupos a
   partir de distancias, sem exemplos com resposta correta.

2. **O que K representa?**
   O numero de clusters escolhido antes da execucao. Aqui K=64 no benchmark.

3. **Por que nao usamos raiz quadrada?**
   `sqrt` preserva a ordem; minimizar a distancia e minimizar seu quadrado dao
   o mesmo label, economizando uma operacao cara.

4. **Como o empate e resolvido?**
   `label` inicia em zero e so muda quando `dist < best`; igualdade conserva o
   menor indice visitado primeiro.

5. **O que acontece com cluster vazio?**
   A atualizacao testa contagem zero e nao escreve, mantendo o centroide antigo.

6. **Qual e o gargalo assintotico?**
   A atribuicao O(I*N*K*D). Somar o ponto escolhido custa O(I*N*D), menor por um
   fator aproximado K.

7. **Por que as iteracoes nao podem rodar juntas?**
   A atribuicao da iteracao seguinte precisa dos centroides atualizados pela
   anterior; haveria dependencia verdadeira e resultado incorreto.

8. **Onde esta o MAP?**
   Na atribuicao independente de cada ponto. OpenMP reparte indices e CUDA
   mapeia threads para pontos.

9. **Onde esta o REDUCE?**
   Na combinacao das somas e contagens por cluster: reduction OpenMP e parciais
   por bloco mais kernels de reducao no CUDA.

10. **Por que `schedule(static)`?**
    Cada ponto faz essencialmente K*D operacoes, logo a carga e uniforme e nao
    justifica o custo de agendamento dinamico.

11. **Por que nao usar `critical` no OpenMP?**
    Serializaria o laco quente. A reducao cria copias privadas e combina apenas
    no final.

12. **labels tem data race?**
    Nao. Cada iteracao i do laco e executada por uma unica thread e escreve
    exclusivamente `labels[i]`.

13. **Por que somas sao double se pontos sao float?**
    Somar centenas de milhares de floats acumula erro. Double aproxima a
    referencia CPU e produziu exatamente a mesma inercia observada.

14. **Por que a GPU exige CC 6.0?**
    O codigo usa `atomicAdd(double)`, suportado diretamente a partir de Pascal
    (compute capability 6.0).

15. **Por que shared memory ajuda?**
    Atomicas de todos os blocos em um unico vetor global gerariam alta disputa.
    Cada bloco disputa apenas seu vetor local e grava um parcial independente.

16. **Por que limitar o grid a oito blocos por SM?**
    Um grid-stride loop ainda cobre N e o limite controla a memoria O(B*K*D)
    dos parciais. Oito ondas fornecem paralelismo suficiente para este kernel.

17. **O que garante a ordem assign -> reduce -> update?**
    Lancamentos no mesmo default stream sao ordenados. Cada kernel seguinte so
    inicia depois do anterior no mesmo stream.

18. **O tempo CUDA inclui transferencias?**
    Sim. `main.c` mede toda `kmeans_run`: alocacoes, HtoD, kernels,
    sincronizacao, DtoH e liberacao. Parsing e inercia final ficam fora em todas.

19. **Por que o `real` e maior que `time_s`?**
    `real` inclui carregar 581 mil linhas, validar CSV, normalizar e calcular
    inercia no host. `time_s` isola o algoritmo conforme o contrato.

20. **Como foi validado?**
    Casos pequenos e bordas foram comparados a uma referencia CPU; o teste CUDA
    rodou na GPU; no Covertype, todas as inercia foram 120313.163175.

21. **O speedup CUDA pode usar os 54 s do i5?**
    Nao. Comparacao controlada exige a mesma maquina. Usamos 147.170/13.068 =
    11.26 no i7/WSL; o benchmark do Arthur e outro experimento.

22. **Por que a eficiencia OpenMP cai apos quatro threads?**
    Ha apenas seis nucleos fisicos; depois entram SMT, reducao, largura de banda,
    fracao sequencial e, em 16/32, oversubscription sobre 12 CPUs logicas.

23. **O que diz a Lei de Amdahl aqui?**
    Mesmo que a atribuicao acelere infinitamente, loader, atualizacao, reducao e
    outros trechos seriais limitam o speedup total.

24. **K-means sempre acha a melhor particao?**
    Nao. Lloyd converge para um minimo local dependente da inicializacao. A seed
    fixa garante reproducibilidade, nao otimalidade global.

25. **Por que normalizar apenas dez colunas?**
    Sao quantitativas com escalas diferentes; sem normalizar dominariam a
    distancia. As 44 restantes ja sao indicadores binarios 0/1.

26. **O que mudaria com convergencia antecipada?**
    Seria preciso detectar labels/centroides inalterados e reduzir essa condicao.
    O trabalho exige comparacao justa, entao todas executam exatamente 50 vezes.

27. **Por que o resultado pode variar nos ultimos bits?**
    Soma de ponto flutuante nao e associativa; reducoes mudam a ordem. Por isso
    se usa tolerancia, embora a inercia registrada tenha sido identica.

28. **O que faria para otimizar mais?**
    Mediria com profiler, avaliaria tiles de centroides, layout para coalescencia,
    menos atomicas e ocupacao. So mudaria precisao com validacao e documentacao;
    nao usaria `fast-math` silenciosamente.

29. **Qual e a limitacao dos labels retornados?**
    Sao os labels usados na ultima atualizacao, antes dos centroides finais. Uma
    reatribuicao extra mudaria custo e semantica; todas as versoes sao iguais.

30. **Qual foi a principal decisao CUDA?**
    Manter dados residentes e usar reducao parcial em shared memory, equilibrando
    corretude FP64, menor disputa global e uma explicacao defensavel.

31. **Como sabemos que o fallback CUDA funciona?**
    `test_cuda` usa K=64 e D=1024: acumuladores e contagens requerem cerca de
    513 KiB, acima da shared memory por bloco, forcam o caminho global e sao
    comparados com a referencia CPU. O caso tambem valida inercia zero.
