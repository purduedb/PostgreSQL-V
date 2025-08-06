# PostgreSQL-V
Vector databases have recently gained significant attention due to the emergence of LLMs. While developing specialized vector databases is interesting, there is a substantial customer base interested in integrated vector databases (that build vector search into existing relational databases like PostgreSQL) for various reasons. However, we observe a substantial performance gap between specialized and integrated vector databases, which raises an interesting research question: _Is it possible to bridge this performance gap_?

In this paper, we introduce PostgreSQL-V, a new system that enables fast vector search in PostgreSQL. Unlike prior work (e.g., pgvector) that inherits legacy overhead by reusing PostgreSQL's page-oriented structure, PostgreSQL-V adopts a novel architectural design that decouples vector indexes from PostgreSQL's core engine. This decoupling offers many benefits, such as directly leveraging native vector index libraries for high performance. However, it also introduces the challenge of index inconsistency, which we address with a lightweight consistency mechanism. Experiments show that PostgreSQL-V achieves performance on par with specialized vector databases and **outperforms pgvector by up to 8.9×** in vector search. To our knowledge, this is the first work to deliver specialized-level performance for vector search in PostgreSQL. We believe its insights can shed light on designing fast vector search in other relational databases, e.g., MySQL and DuckDB.

## Architecture Overview
![](figures/CIDR_arch_v10.png)
## Experiment Overview
**SIFT10M**
![](figures/SIFT10M.png)
**DEEP10M**
![](figures/DEEP10M.png)
