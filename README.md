# PostgreSQL-V
**Open-Source Vector Similarity Search for PostgreSQL**

## Introduction
**PostgreSQL-V** is built on top of the open-source project [pgvector](https://github.com/pgvector/pgvector), extending PostgreSQL with high-performance vector similarity search.  

Unlike prior extensions such as **Pase**, **pgvector**, and **pgvectorscale**, which inherit legacy overhead from PostgreSQL’s page-oriented storage layer, PostgreSQL-V introduces a **novel decoupled architecture** that separates vector indexes from PostgreSQL’s core engine.  

This architectural shift enables PostgreSQL-V to:
- **Integrate native vector index libraries** (e.g., *Faiss*, *hnswlib*) in a pluggable fashion, achieving state-of-the-art performance while keeping PostgreSQL’s simplicity.  
- **Adopt an LSM-based framework** for efficient index updates and improved concurrency for both reads and writes. 

Besides, PostgreSQL-V preserves full SQL compatibility and maintains PostgreSQL's transactional **ACID** properties

## Installation
### 1. Install PostgreSQL
Follow the official [PostgreSQL installation guide](https://www.postgresql.org/download/) for your operating system.

### 2. Build and Install the Extension
```bash
cd postgresqlv
vim ./install_pgvector.sh
```
Locate the line starting with:
```
PG_CONFIG=
```
and update it to point to your local pg_config path. Then run:
```bash
./install_pgvector.sh
```

## Getting Started
PostgreSQL-V is fully compatible with pgvector.
You can follow pgvector's [user tutorial](https://github.com/pgvector/pgvector) to get started with basic usage and examples.

## Architecture Overview
The following figure illustrates the high-level architecture of PostgreSQL-V.

(Details of the design will be provided in our upcoming CIDR paper; the link will be added here once available.)
![](figures/CIDR_arch_v10.png)

## Experiment Overview
**SIFT10M**
![](figures/SIFT10M.png)
**DEEP10M**
![](figures/DEEP10M.png)

## Note
### ⚠️ Prototype Notice:
This branch contains the prototype version used in our CIDR submission to validate the proposed design.

The full implementation is under development in the main branch. 🚧