# Report Executivo: Personalização do Carrossel de Produtos Financeiros

## 1. Overview e Abordagem Final

O objetivo deste projeto foi desenvolver um sistema de recomendação para personalizar a ordem de 20 produtos em um carrossel horizontal para ~50.000 clientes, focando na otimização das 5 primeiras posições (visíveis sem scroll) para maximizar a contratação e a receita.

Abordagem Escolhida: foi implementado um classificador XGBoost "point wise". Esta técnica foi selecionada por obter as melhores métricas primárias de Precision@5 e NDCG@5 dentre os modelos desafiantes, mesmo quando comparada a estratégias dedicadas para recomendação como o Lambda MART.

## 2. Principais Descobertas da EDA (analise_exploratoria.ipynb)

A análise exploratória revelou características interessantes para problema de recomendação:

* Sazonalidade: produtos como previdência apresentam ciclos anuais fortes, influenciando  a propensão de contratação em meses específicos (safras de final de ano).

* Comportamento Pós-Clique: identificou-se que a propensão de contratação aumenta significativamente após o primeiro clique, sugerindo que sinais de navegação em tempo real são preditores valiosos.

* Força do Baseline: as regras atuais baseadas em segmento (Básico, Intermediário, Premium) são competitivas, estabelecendo um patamar alto para ganho incremental (lift).

* Poucas interações: é um problema de eventos raros, a maioria das recomendações não tem interação do cliente, o que cria desafios para a modelagem.

## 3. Modelagem e Resultados (modelos_desafiantes.ipynb e resultados.ipynb)

Os experimentos focaram em superar o baseline estático através de modelos de boosting:

* Impacto da Safra: a adição da informação temporal ("safra") gerou o maior lift no NDCG@5, embora sua utilização exija cautela para garantir a generalização em períodos futuros. **Não foi utlizado** na predição final conforme recomendado por e-mail.

* Cold start de clientes: clientes novos não parecem impactar tanto nos modelos testados e nem nos baselines.

* Otimização de Negócio: foram testadas estratégias de pesos de amostra baseados na receita média dos produtos para alinhar o modelo ao objetivo financeiro do banco.

* Desempenho: os modelos ficaram muito próximos do baseline, mesmo sem grandes alterações. O modelo atual (XGBoost) apresentou tendências de overfitting durante a busca de hiperparâmetros, exigindo maior refinamento no tratamento do desequilíbrio de classes.

* Insights: a efetividade do baseline indica caminhos possíveis de evolução. Uso de modelos mais simples e eficientes podem ser a resposta, abordagens alternativas podem ser o caminho correto.

## 4. Arquitetura de Produção (arquitetura_producao.ipynb)

A arquitetura ideal depende dos avanços que serão feitos com modelos, especialmente em dados. Para os dados atuais, uma solução batch é adequda. Para o futuro, entendo que o caminho ideal é uma estratégida de predição NRT (Near Real Time) com dispnonbilização em RT.

Independente da arquitetura de predizer e servir, alguns aspectos chaves precisam ser tratados:

* Monitoramento: implementação de detecção de Model Drift e Feature Drift, garantindo que a necessidade de re-treino e possíveis bugs sejam rapidamente identificados.

* Proposta de teste A/B: uma solução de teste A/B depende de muitas outras partes, mas é possível implementar uma solução simples desde a arquitetura mais básica proposta.

* Cold start de produtos: os modelos baseline e propostos não tratam esse problema diretamente. Uma solução de engenharia possível, seria forçar a recomendação de determinados produtos independente do modelo.

5. Limitações e Próximos Passos

Limitação: os modelos propostos não tratam a questão de novos produtos e apresentaram restultados similares ao baseline nos testes.

Evolução: explorar dados que possam ser relevantes; evoluir os modelos desenvolvidos para ajuste fine de hiperparâmetros e transformação de dados; uso de modelos que possam ter melhor poder de generalização e lidar com limitações da solução atual.
