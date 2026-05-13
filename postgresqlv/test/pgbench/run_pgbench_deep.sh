#!/bin/bash

# warmup
echo "Warmup..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_queries.sql -c 1 -j 1 -T 120 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 100, probes = 2, conurrency - 1..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_100.sql -c 1 -j 1 -T 60 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 200, probes = 3, conurrency - 1..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_200.sql -c 1 -j 1 -T 60 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 300, probes = 5, conurrency - 1..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_300.sql -c 1 -j 1 -T 60 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 400, probes = 10, conurrency - 1..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_400.sql -c 1 -j 1 -T 60 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 500, probes = 20, conurrency - 1..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_500.sql -c 1 -j 1 -T 60 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 600, probes = 40, conurrency - 1..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_600.sql -c 1 -j 1 -T 60 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 700, probes = 60, conurrency - 1..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_700.sql -c 1 -j 1 -T 60 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 800, probes = 80, conurrency - 1..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_800.sql -c 1 -j 1 -T 60 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 900, probes = 100, conurrency - 1..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_900.sql -c 1 -j 1 -T 60 -h 127.0.0.1 -p 5435 postgres

# echo "Run queries with ef_search = 1000, probes = 120, conurrency - 1..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_1000.sql -c 1 -j 1 -T 600 -h 127.0.0.1 -p 5435 postgres


echo "Run queries with ef_search = 100, probes = 2, conurrency - 2..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_100.sql -c 2 -j 2 -T 30 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 200, probes = 3, conurrency - 2..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_200.sql -c 2 -j 2 -T 30 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 300, probes = 5, conurrency - 2..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_300.sql -c 2 -j 2 -T 30 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 400, probes = 10, conurrency - 2..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_400.sql -c 2 -j 2 -T 60 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 500, probes = 20, conurrency - 2..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_500.sql -c 2 -j 2 -T 60 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 600, probes = 40, conurrency - 2..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_600.sql -c 2 -j 2 -T 60 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 700, probes = 60, conurrency - 2..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_700.sql -c 2 -j 2 -T 60 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 800, probes = 80, conurrency - 2..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_800.sql -c 2 -j 2 -T 60 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 900, probes = 100, conurrency - 2..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_900.sql -c 2 -j 2 -T 60 -h 127.0.0.1 -p 5435 postgres

# echo "Run queries with ef_search = 1000, probes = 120, conurrency - 2..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_1000.sql -c 2 -j 2 -T 300 -h 127.0.0.1 -p 5435 postgres


echo "Run queries with ef_search = 100, probes = 2, conurrency - 4..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_100.sql -c 4 -j 4 -T 30 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 200, probes = 3, conurrency - 4..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_200.sql -c 4 -j 4 -T 30 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 300, probes = 5, conurrency - 4..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_300.sql -c 4 -j 4 -T 30 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 400, probes = 8, conurrency - 4..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_400.sql -c 4 -j 4 -T 30 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 500, probes = 20, conurrency - 4..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_500.sql -c 4 -j 4 -T 30 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 600, probes = 40, conurrency - 4..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_600.sql -c 4 -j 4 -T 30 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 700, probes = 60, conurrency - 4..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_700.sql -c 4 -j 4 -T 30 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 800, probes = 80, conurrency - 4..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_800.sql -c 4 -j 4 -T 30 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 900, probes = 100, conurrency - 4..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_900.sql -c 4 -j 4 -T 30 -h 127.0.0.1 -p 5435 postgres

# echo "Run queries with ef_search = 1000, probes = 120, conurrency - 4..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_1000.sql -c 4 -j 4 -T 30 -h 127.0.0.1 -p 5435 postgres


echo "Run queries with ef_search = 100, probes = 2, conurrency - 8..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_100.sql -c 8 -j 8 -T 30 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 200, probes = 3, conurrency - 8..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_200.sql -c 8 -j 8 -T 30 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 300, probes = 5, conurrency - 8..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_300.sql -c 8 -j 8 -T 30 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 400, probes = 10, conurrency - 8..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_400.sql -c 8 -j 8 -T 30 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 500, probes = 20, conurrency - 8..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_500.sql -c 8 -j 8 -T 30 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 600, probes = 40, conurrency - 8..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_600.sql -c 8 -j 8 -T 30 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 700, probes = 60, conurrency - 8..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_700.sql -c 8 -j 8 -T 30 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 800, probes = 80, conurrency - 8..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_800.sql -c 8 -j 8 -T 30 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 900, probes = 100, conurrency - 8..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_900.sql -c 8 -j 8 -T 30 -h 127.0.0.1 -p 5435 postgres

# echo "Run queries with ef_search = 1000, probes = 120, conurrency - 8..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_1000.sql -c 8 -j 8 -T 30 -h 127.0.0.1 -p 5435 postgres


echo "Run queries with ef_search = 100, probes = 2, conurrency - 16..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_100.sql -c 16 -j 8 -T 30 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 200, probes = 3, conurrency - 16..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_200.sql -c 16 -j 8 -T 30 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 300, probes = 5, conurrency - 16..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_300.sql -c 16 -j 8 -T 30 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 400, probes = 10, conurrency - 16..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_400.sql -c 16 -j 8 -T 30 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 500, probes = 20, conurrency - 16..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_500.sql -c 16 -j 8 -T 30 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 600, probes = 40, conurrency - 16..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_600.sql -c 16 -j 8 -T 30 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 700, probes = 60, conurrency - 16..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_700.sql -c 16 -j 8 -T 10 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 800, probes = 80, conurrency - 16..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_800.sql -c 16 -j 8 -T 30 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 900, probes = 100, conurrency - 16..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_900.sql -c 16 -j 8 -T 30 -h 127.0.0.1 -p 5435 postgres

# echo "Run queries with ef_search = 1000, probes = 120, conurrency - 16..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_1000.sql -c 16 -j 8 -T 10 -h 127.0.0.1 -p 5435 postgres


echo "Run queries with ef_search = 100, probes = 2, conurrency - 32..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_100.sql -c 32 -j 8 -T 30 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 200, probes = 3, conurrency - 32..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_200.sql -c 32 -j 8 -T 30 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 300, probes = 5, conurrency - 32..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_300.sql -c 32 -j 8 -T 30 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 400, probes = 10, conurrency - 32..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_400.sql -c 32 -j 8 -T 30 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 500, probes = 20, conurrency - 32..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_500.sql -c 32 -j 8 -T 30 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 600, probes = 40, conurrency - 32..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_600.sql -c 32 -j 8 -T 30 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 700, probes = 60, conurrency - 32..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_700.sql -c 32 -j 8 -T 30 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 800, probes = 80, conurrency - 32..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_800.sql -c 32 -j 8 -T 30 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 900, probes = 100, conurrency - 32..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_900.sql -c 32 -j 8 -T 30 -h 127.0.0.1 -p 5435 postgres

# echo "Run queries with ef_search = 1000, probes = 120, conurrency - 32..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/deep_query_scripts/deep_queries_random_1000.sql -c 32 -j 8 -T 120 -h 127.0.0.1 -p 5435 postgres