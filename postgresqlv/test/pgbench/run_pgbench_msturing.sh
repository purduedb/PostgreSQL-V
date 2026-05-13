#!/bin/bash

# warmup
echo "Warmup..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_queries.sql -c 1 -j 1 -T 120 -h 127.0.0.1 -p 5434 postgres

# echo "Run queries with ef_search = 100, conurrency - 1..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_100.sql -c 1 -j 1 -T 60 -h 127.0.0.1 -p 5434 postgres

# echo "Run queries with ef_search = 200, conurrency - 1..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_200.sql -c 1 -j 1 -T 60 -h 127.0.0.1 -p 5434 postgres

# echo "Run queries with ef_search = 300, conurrency - 1..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_300.sql -c 1 -j 1 -T 60 -h 127.0.0.1 -p 5434 postgres

# echo "Run queries with ef_search = 400, conurrency - 1..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_400.sql -c 1 -j 1 -T 60 -h 127.0.0.1 -p 5434 postgres

# echo "Run queries with ef_search = 500, conurrency - 1..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_500.sql -c 1 -j 1 -T 60 -h 127.0.0.1 -p 5434 postgres

# echo "Run queries with ef_search = 600, conurrency - 1..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_600.sql -c 1 -j 1 -T 60 -h 127.0.0.1 -p 5434 postgres

# echo "Run queries with ef_search = 700, conurrency - 1..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_700.sql -c 1 -j 1 -T 60 -h 127.0.0.1 -p 5434 postgres

# echo "Run queries with ef_search = 800, conurrency - 1..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_800.sql -c 1 -j 1 -T 60 -h 127.0.0.1 -p 5434 postgres

# echo "Run queries with ef_search = 900, conurrency - 1..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_900.sql -c 1 -j 1 -T 60 -h 127.0.0.1 -p 5434 postgres

# echo "Run queries with ef_search = 1000, conurrency - 1..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_1000.sql -c 1 -j 1 -T 60 -h 127.0.0.1 -p 5434 postgres


# echo "Run queries with ef_search = 100, conurrency - 2..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_100.sql -c 2 -j 2 -T 30 -h 127.0.0.1 -p 5434 postgres

# echo "Run queries with ef_search = 200, conurrency - 2..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_200.sql -c 2 -j 2 -T 30 -h 127.0.0.1 -p 5434 postgres

# echo "Run queries with ef_search = 300, conurrency - 2..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_300.sql -c 2 -j 2 -T 30 -h 127.0.0.1 -p 5434 postgres

# echo "Run queries with ef_search = 400, conurrency - 2..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_400.sql -c 2 -j 2 -T 30 -h 127.0.0.1 -p 5434 postgres

# echo "Run queries with ef_search = 500, conurrency - 2..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_500.sql -c 2 -j 2 -T 30 -h 127.0.0.1 -p 5434 postgres

# echo "Run queries with ef_search = 600, conurrency - 2..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_600.sql -c 2 -j 2 -T 30 -h 127.0.0.1 -p 5434 postgres

# echo "Run queries with ef_search = 700, conurrency - 2..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_700.sql -c 2 -j 2 -T 30 -h 127.0.0.1 -p 5434 postgres

# echo "Run queries with ef_search = 800, conurrency - 2..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_800.sql -c 2 -j 2 -T 30 -h 127.0.0.1 -p 5434 postgres

# echo "Run queries with ef_search = 900, conurrency - 2..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_900.sql -c 2 -j 2 -T 30 -h 127.0.0.1 -p 5434 postgres

# echo "Run queries with ef_search = 1000, conurrency - 2..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_1000.sql -c 2 -j 2 -T 30 -h 127.0.0.1 -p 5434 postgres


# echo "Run queries with ef_search = 100, conurrency - 4..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_100.sql -c 4 -j 4 -T 30 -h 127.0.0.1 -p 5434 postgres

# echo "Run queries with ef_search = 200, conurrency - 4..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_200.sql -c 4 -j 4 -T 30 -h 127.0.0.1 -p 5434 postgres

# echo "Run queries with ef_search = 300, conurrency - 4..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_300.sql -c 4 -j 4 -T 30 -h 127.0.0.1 -p 5434 postgres

# echo "Run queries with ef_search = 400, conurrency - 4..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_400.sql -c 4 -j 4 -T 30 -h 127.0.0.1 -p 5434 postgres

# echo "Run queries with ef_search = 500, conurrency - 4..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_500.sql -c 4 -j 4 -T 30 -h 127.0.0.1 -p 5434 postgres

# echo "Run queries with ef_search = 600, conurrency - 4..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_600.sql -c 4 -j 4 -T 30 -h 127.0.0.1 -p 5434 postgres

# echo "Run queries with ef_search = 700, conurrency - 4..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_700.sql -c 4 -j 4 -T 30 -h 127.0.0.1 -p 5434 postgres

# echo "Run queries with ef_search = 800, conurrency - 4..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_800.sql -c 4 -j 4 -T 30 -h 127.0.0.1 -p 5434 postgres

# echo "Run queries with ef_search = 900, conurrency - 4..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_900.sql -c 4 -j 4 -T 30 -h 127.0.0.1 -p 5434 postgres

# echo "Run queries with ef_search = 1000, conurrency - 4..."
# pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_1000.sql -c 4 -j 4 -T 30 -h 127.0.0.1 -p 5434 postgres


echo "Run queries with ef_search = 100, conurrency - 8..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_100.sql -c 8 -j 8 -T 30 -h 127.0.0.1 -p 5434 postgres

echo "Run queries with ef_search = 200, conurrency - 8..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_200.sql -c 8 -j 8 -T 30 -h 127.0.0.1 -p 5434 postgres

echo "Run queries with ef_search = 300, conurrency - 8..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_300.sql -c 8 -j 8 -T 30 -h 127.0.0.1 -p 5434 postgres

echo "Run queries with ef_search = 400, conurrency - 8..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_400.sql -c 8 -j 8 -T 30 -h 127.0.0.1 -p 5434 postgres

echo "Run queries with ef_search = 500, conurrency - 8..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_500.sql -c 8 -j 8 -T 30 -h 127.0.0.1 -p 5434 postgres

echo "Run queries with ef_search = 600, conurrency - 8..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_600.sql -c 8 -j 8 -T 30 -h 127.0.0.1 -p 5434 postgres

echo "Run queries with ef_search = 700, conurrency - 8..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_700.sql -c 8 -j 8 -T 30 -h 127.0.0.1 -p 5434 postgres

echo "Run queries with ef_search = 800, conurrency - 8..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_800.sql -c 8 -j 8 -T 30 -h 127.0.0.1 -p 5434 postgres

echo "Run queries with ef_search = 900, conurrency - 8..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_900.sql -c 8 -j 8 -T 30 -h 127.0.0.1 -p 5434 postgres

echo "Run queries with ef_search = 1000, conurrency - 8..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_1000.sql -c 8 -j 8 -T 30 -h 127.0.0.1 -p 5434 postgres


echo "Run queries with ef_search = 100, conurrency - 16..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_100.sql -c 16 -j 8 -T 10 -h 127.0.0.1 -p 5434 postgres

echo "Run queries with ef_search = 200, conurrency - 16..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_200.sql -c 16 -j 8 -T 10 -h 127.0.0.1 -p 5434 postgres

echo "Run queries with ef_search = 300, conurrency - 16..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_300.sql -c 16 -j 8 -T 10 -h 127.0.0.1 -p 5434 postgres

echo "Run queries with ef_search = 400, conurrency - 16..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_400.sql -c 16 -j 8 -T 10 -h 127.0.0.1 -p 5434 postgres

echo "Run queries with ef_search = 500, conurrency - 16..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_500.sql -c 16 -j 8 -T 10 -h 127.0.0.1 -p 5434 postgres

echo "Run queries with ef_search = 600, conurrency - 16..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_600.sql -c 16 -j 8 -T 10 -h 127.0.0.1 -p 5434 postgres

echo "Run queries with ef_search = 700, conurrency - 16..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_700.sql -c 16 -j 8 -T 10 -h 127.0.0.1 -p 5434 postgres

echo "Run queries with ef_search = 800, conurrency - 16..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_800.sql -c 16 -j 8 -T 10 -h 127.0.0.1 -p 5434 postgres

echo "Run queries with ef_search = 900, conurrency - 16..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_900.sql -c 16 -j 8 -T 10 -h 127.0.0.1 -p 5434 postgres

echo "Run queries with ef_search = 1000, conurrency - 16..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_1000.sql -c 16 -j 8 -T 10 -h 127.0.0.1 -p 5434 postgres


echo "Run queries with ef_search = 100, conurrency - 32..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_100.sql -c 32 -j 8 -T 10 -h 127.0.0.1 -p 5434 postgres

echo "Run queries with ef_search = 200, conurrency - 32..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_200.sql -c 32 -j 8 -T 10 -h 127.0.0.1 -p 5434 postgres

echo "Run queries with ef_search = 300, conurrency - 32..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_300.sql -c 32 -j 8 -T 10 -h 127.0.0.1 -p 5434 postgres

echo "Run queries with ef_search = 400, conurrency - 32..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_400.sql -c 32 -j 8 -T 10 -h 127.0.0.1 -p 5434 postgres

echo "Run queries with ef_search = 500, conurrency - 32..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_500.sql -c 32 -j 8 -T 10 -h 127.0.0.1 -p 5434 postgres

echo "Run queries with ef_search = 600, conurrency - 32..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_600.sql -c 32 -j 8 -T 10 -h 127.0.0.1 -p 5434 postgres

echo "Run queries with ef_search = 700, conurrency - 32..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_700.sql -c 32 -j 8 -T 10 -h 127.0.0.1 -p 5434 postgres

echo "Run queries with ef_search = 800, conurrency - 32..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_800.sql -c 32 -j 8 -T 10 -h 127.0.0.1 -p 5434 postgres

echo "Run queries with ef_search = 900, conurrency - 32..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_900.sql -c 32 -j 8 -T 10 -h 127.0.0.1 -p 5434 postgres

echo "Run queries with ef_search = 1000, conurrency - 32..."
pgbench -n -f /home/liu4127/postgresql/decoupled_pgvector/pgvector/test/pgbench/msturing_query_scripts/msturing_queries_random_1000.sql -c 32 -j 8 -T 10 -h 127.0.0.1 -p 5434 postgres