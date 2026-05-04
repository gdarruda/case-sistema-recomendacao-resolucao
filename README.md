# Report Executivo: Personalização do Carrossel de Produtos Financeiros

## 1. Overview e Abordagem Final

O objetivo deste projeto foi desenvolver um sistema de recomendação para personalizar a ordem de 20 produtos em um carrossel horizontal para ~50.000 clientes, focando na otimização das 5 primeiras posições (visíveis sem scroll) para maximizar a contratação e a receita.

Abordagem Escolhida: foi implementado um classificador XGBoost "point wise". Esta técnica foi selecionada por obter as melhores métricas primárias de Precision@5 e NDCG@5 dentre os modelos desafiantes, mesmo quando comparada a estratégias dedicadas para recomendação como o Lambda MART.

## 2. Principais Descobertas da EDA (analise_exploratoria.ipynb)

A análise exploratória revelou características interessantes para problema de recomendação:

* Sazonalidade: produtos como previdência apresentam ciclos anuais fortes, influenciando  a propensão de contratação em meses específicos (safras de final de ano).

* Comportamento Pós-Clique: identificou-se que a propensão de contratação muda após o primeiro clique, sugerindo que sinais de navegação em tempo real são preditores valiosos.

* Força do Baseline: as regras atuais baseadas em segmento (Básico, Intermediário, Premium) são competitivas, estabelecendo um patamar alto para ganho incremental (lift).

* Poucas interações: é um problema de eventos raros, a maioria das recomendações não tem interação do cliente, o que cria desafios para a modelagem.

## 3. Modelagem e Resultados (modelos_desafiantes.ipynb, modelos_desafiantes_mes.ipynb e resultados.ipynb)

Os experimentos focaram em superar o baseline estático e extrair insights. Foram experimentados modelos de boosting para agilizar a prototipação:

* Impacto da Safra: a adição da informação temporal ("safra") gerou o maior lift no NDCG@5, embora sua utilização exija cautela para garantir a generalização em períodos futuros. **Não foi utlizado** na predição final conforme recomendado por e-mail.

* Cold start de clientes: clientes novos não parecem impactar tanto nos modelos testados e nem nos baselines.

* Otimização de Negócio: foram testadas estratégias de pesos de amostra baseados na receita média dos produtos para alinhar o modelo ao objetivo financeiro do banco.

* Uso de Lambda MART: foi testado o uso de Lambda MART, para otimizar o modelo diretamente para a métrica NDCG. Entretanto, foram encontrados desafios para adaptar o conjunto de dados para o formato ideal.

* Otimização: os modelos desafiantes ficaram muito próximos do baseline. O modelo escolhido apresentou tendências de overfitting durante a busca de hiperparâmetros, indicando espaço para uma otimização mais cuidadosa.

* Insights: a efetividade do baseline indica caminhos possíveis de evolução. Podemos aprimorar o baseline em combinação com uma estratégia de contextual bandits e explorar outras variáveis que possam agregar para os modelos mais complexos.

## 4. Arquitetura de Produção (arquitetura_producao.ipynb)

A arquitetura ideal depende dos avanços que serão feitos com modelos, especialmente na natureza dos dados utilizados. Para o cenário atual, uma solução batch é adequada. Para o futuro, entendo que o caminho ideal é uma estratégida de predição NRT (Near Real Time) com dispnonbilização em RT.

Independente da arquitetura de predizer e servir, alguns aspectos chaves precisam ser tratados:

* Monitoramento: implementação de detecção de Model Drift e Feature Drift, garantindo que a necessidade de re-treino e possíveis bugs sejam rapidamente identificados.

* Proposta de teste A/B: uma solução de teste A/B depende de muitas outras partes, mas é possível implementar uma solução simples desde a arquitetura mais básica proposta.

* Cold start de produtos: os modelos baseline e propostos não tratam esse problema diretamente. Uma solução de engenharia possível, seria forçar a recomendação de determinados produtos independente do modelo.

## 5. Limitações e Próximos Passos

Os modelos propostos não obtiveram bons resultados, mas existe uma perspectiva de explorar melhor o ajuste fino de parâmetros e tratamento de dados. Entretanto, existem limitações conceituais como a questão de lidar com produtos novos em uma solução pointwise como a proposta. Nesse contexto, vejo os seguintes passos:

* explorar dados que possam ser relevantes para predição, especialmente pensando em interação com os produtos e canais (aplicativo, chats, transações);

* evoluir os modelos desenvolvidos para ajuste fino de hiperparâmetros, transformação de dados e validações mais robustas;

* explorar alternativas que lidem com as limitações do baseline, como Contextual Bandits para novos produtos;

* experimentar modelos com abordagens complementares, como filtros basedo em  conteúdo e redes neurais.
