-- pgbench script for vector similarity search
-- pgbench handles transactions automatically, so no BEGIN/COMMIT needed
-- The script will be executed multiple times based on -c (clients) and -t (transactions) parameters

-- Pick a random query id between 1 and 1000
\set qid random(1, 1000)

SET hnsw.ef_search = 400;

-- Run similarity search using scalar subquery in ORDER BY
-- This allows the planner to use the vector index efficiently
SELECT id
FROM pg_vectorscale_collection
ORDER BY embedding <-> (SELECT v FROM cohere_queries WHERE id = :qid)
LIMIT 100;

