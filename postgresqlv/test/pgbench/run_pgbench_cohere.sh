#!/bin/bash

# warmup
echo "Warmup..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_queries.sql -c 1 -j 1 -T 120 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 100, conurrency - 1..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_100.sql -c 1 -j 1 -T 60 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 200, conurrency - 1..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_200.sql -c 1 -j 1 -T 60 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 250, conurrency - 1..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_250.sql -c 1 -j 1 -T 60 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 300, conurrency - 1..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_300.sql -c 1 -j 1 -T 60 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 350, conurrency - 1..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_350.sql -c 1 -j 1 -T 60 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 400, conurrency - 1..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_400.sql -c 1 -j 1 -T 60 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 500, conurrency - 1..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_500.sql -c 1 -j 1 -T 60 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 600, conurrency - 1..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_600.sql -c 1 -j 1 -T 60 -h 127.0.0.1 -p 5435 postgres


# echo "Run queries with ef_search = 100, conurrency - 2..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_100.sql -c 2 -j 2 -T 30 -h 127.0.0.1 -p 5435 postgres

# echo "Run queries with ef_search = 200, conurrency - 2..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_200.sql -c 2 -j 2 -T 30 -h 127.0.0.1 -p 5435 postgres

# echo "Run queries with ef_search = 250, conurrency - 2..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_250.sql -c 2 -j 2 -T 30 -h 127.0.0.1 -p 5435 postgres

# echo "Run queries with ef_search = 300, conurrency - 2..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_300.sql -c 2 -j 2 -T 30 -h 127.0.0.1 -p 5435 postgres

# echo "Run queries with ef_search = 350, conurrency - 2..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_350.sql -c 2 -j 2 -T 60 -h 127.0.0.1 -p 5435 postgres

# echo "Run queries with ef_search = 400, conurrency - 2..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_400.sql -c 2 -j 2 -T 60 -h 127.0.0.1 -p 5435 postgres

# echo "Run queries with ef_search = 500, conurrency - 2..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_500.sql -c 2 -j 2 -T 60 -h 127.0.0.1 -p 5435 postgres

# echo "Run queries with ef_search = 600, conurrency - 2..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_600.sql -c 2 -j 2 -T 60 -h 127.0.0.1 -p 5435 postgres


# echo "Run queries with ef_search = 100, conurrency - 4..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_100.sql -c 4 -j 4 -T 30 -h 127.0.0.1 -p 5435 postgres

# echo "Run queries with ef_search = 200, conurrency - 4..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_200.sql -c 4 -j 4 -T 30 -h 127.0.0.1 -p 5435 postgres

# echo "Run queries with ef_search = 250, conurrency - 4..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_250.sql -c 4 -j 4 -T 30 -h 127.0.0.1 -p 5435 postgres

# echo "Run queries with ef_search = 300, conurrency - 4..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_300.sql -c 4 -j 4 -T 30 -h 127.0.0.1 -p 5435 postgres

# echo "Run queries with ef_search = 350, conurrency - 4..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_350.sql -c 4 -j 4 -T 30 -h 127.0.0.1 -p 5435 postgres

# echo "Run queries with ef_search = 400, conurrency - 4..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_400.sql -c 4 -j 4 -T 30 -h 127.0.0.1 -p 5435 postgres

# echo "Run queries with ef_search = 500, conurrency - 4..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_500.sql -c 4 -j 4 -T 60 -h 127.0.0.1 -p 5435 postgres

# echo "Run queries with ef_search = 600, conurrency - 4..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_600.sql -c 4 -j 4 -T 60 -h 127.0.0.1 -p 5435 postgres


# echo "Run queries with ef_search = 100, conurrency - 8..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_100.sql -c 8 -j 8 -T 30 -h 127.0.0.1 -p 5435 postgres

# echo "Run queries with ef_search = 200, conurrency - 8..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_200.sql -c 8 -j 8 -T 30 -h 127.0.0.1 -p 5435 postgres

# echo "Run queries with ef_search = 250, conurrency - 8..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_250.sql -c 8 -j 8 -T 30 -h 127.0.0.1 -p 5435 postgres

# echo "Run queries with ef_search = 300, conurrency - 8..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_300.sql -c 8 -j 8 -T 30 -h 127.0.0.1 -p 5435 postgres

# echo "Run queries with ef_search = 350, conurrency - 8..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_350.sql -c 8 -j 8 -T 60 -h 127.0.0.1 -p 5435 postgres

# echo "Run queries with ef_search = 400, conurrency - 8..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_400.sql -c 8 -j 8 -T 60 -h 127.0.0.1 -p 5435 postgres

# echo "Run queries with ef_search = 500, conurrency - 8..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_500.sql -c 8 -j 8 -T 60 -h 127.0.0.1 -p 5435 postgres

# echo "Run queries with ef_search = 600, conurrency - 8..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_600.sql -c 8 -j 8 -T 60 -h 127.0.0.1 -p 5435 postgres


# echo "Run queries with ef_search = 100, conurrency - 16..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_100.sql -c 16 -j 16 -T 30 -h 127.0.0.1 -p 5435 postgres

# echo "Run queries with ef_search = 200, conurrency - 16..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_200.sql -c 16 -j 16 -T 30 -h 127.0.0.1 -p 5435 postgres

# echo "Run queries with ef_search = 250, conurrency - 16..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_250.sql -c 16 -j 16 -T 30 -h 127.0.0.1 -p 5435 postgres

# echo "Run queries with ef_search = 300, conurrency - 16..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_300.sql -c 16 -j 16 -T 30 -h 127.0.0.1 -p 5435 postgres

# echo "Run queries with ef_search = 350, conurrency - 16..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_350.sql -c 16 -j 16 -T 30 -h 127.0.0.1 -p 5435 postgres

# echo "Run queries with ef_search = 400, conurrency - 16..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_400.sql -c 16 -j 16 -T 30 -h 127.0.0.1 -p 5435 postgres

# echo "Run queries with ef_search = 500, conurrency - 16..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_500.sql -c 16 -j 16 -T 30 -h 127.0.0.1 -p 5435 postgres

# echo "Run queries with ef_search = 600, conurrency - 16..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_600.sql -c 16 -j 16 -T 30 -h 127.0.0.1 -p 5435 postgres


echo "Run queries with ef_search = 100, conurrency - 32..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_100.sql -c 32 -j 32 -T 180 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 200, conurrency - 32..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_200.sql -c 32 -j 32 -T 180 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 250, conurrency - 32..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_250.sql -c 32 -j 32 -T 180 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 300, conurrency - 32..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_300.sql -c 32 -j 32 -T 180 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 350, conurrency - 32..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_350.sql -c 32 -j 32 -T 180 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 400, conurrency - 32..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_400.sql -c 32 -j 32 -T 180 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 500, conurrency - 32..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_500.sql -c 32 -j 32 -T 180 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 600, conurrency - 32..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_600.sql -c 32 -j 32 -T 180 -h 127.0.0.1 -p 5435 postgres



echo "Run queries with ef_search = 100, conurrency - 1..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_100.sql -c 1 -j 1 -T 120 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 200, conurrency - 1..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_200.sql -c 1 -j 1 -T 120 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 250, conurrency - 1..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_250.sql -c 1 -j 1 -T 120 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 300, conurrency - 1..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_300.sql -c 1 -j 1 -T 120 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 350, conurrency - 1..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_350.sql -c 1 -j 1 -T 120 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 400, conurrency - 1..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_400.sql -c 1 -j 1 -T 120 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 500, conurrency - 1..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_500.sql -c 1 -j 1 -T 120 -h 127.0.0.1 -p 5435 postgres

echo "Run queries with ef_search = 600, conurrency - 1..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/cohere_query_scripts/cohere_queries_random_600.sql -c 1 -j 1 -T 120 -h 127.0.0.1 -p 5435 postgres