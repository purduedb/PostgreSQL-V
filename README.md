# PostgreSQL-V
**Open-Source Vector Similarity Search for PostgreSQL**

## Note
⚠️The `main` branch is currently under active development.  

For an executable prototype, please switch to the [`cidr` branch](https://github.com/purduedb/PostgreSQL-V/tree/cidr), which contains the version used in our CIDR submission to validate the proposed design.

Installation instructions and usage details can be found in the `README` of the `cidr` branch

## Introduction
**PostgreSQL-V** is built on top of the open-source project [pgvector](https://github.com/pgvector/pgvector), extending PostgreSQL with high-performance vector similarity search.  

Unlike prior extensions such as **Pase**, **pgvector**, and **pgvectorscale**, which inherit legacy overhead from PostgreSQL’s page-oriented storage layer, PostgreSQL-V introduces a **novel decoupled architecture** that separates vector indexes from PostgreSQL’s core engine.  

This architectural shift enables PostgreSQL-V to:
- **Integrate native vector index libraries** (e.g., *Faiss*, *hnswlib*) in a pluggable fashion, achieving state-of-the-art performance while keeping PostgreSQL’s simplicity.  
- **Adopt an LSM-based framework** for efficient index updates and improved concurrency for both reads and writes. 

Besides, PostgreSQL-V preserves full SQL compatibility and maintains PostgreSQL's transactional **ACID** properties