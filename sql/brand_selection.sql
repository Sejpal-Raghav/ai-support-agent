-- Full-data candidate query. The Python entrypoint binds $csv_path.
WITH tweets AS (
    SELECT
        trim(CAST(tweet_id AS VARCHAR)) AS tweet_id,
        trim(CAST(author_id AS VARCHAR)) AS author_id,
        lower(trim(CAST(inbound AS VARCHAR))) = 'true' AS inbound,
        try_strptime(created_at, '%a %b %d %H:%M:%S %z %Y') AS created_at,
        CAST(text AS VARCHAR) AS text,
        nullif(trim(CAST(in_response_to_tweet_id AS VARCHAR)), '') AS parent_id
    FROM read_csv($csv_path, header=true, all_varchar=true, nullstr=['', 'NA', 'nan'])
    WHERE tweet_id IS NOT NULL AND author_id IS NOT NULL AND text IS NOT NULL
),
direct_pairs AS (
    SELECT
        b.author_id AS brand_id,
        c.tweet_id AS customer_tweet_id,
        b.tweet_id AS brand_tweet_id,
        c.created_at AS customer_at,
        b.created_at AS reply_at
    FROM tweets b
    INNER JOIN tweets c ON b.parent_id = c.tweet_id
    WHERE b.inbound = false AND c.inbound = true AND b.created_at >= c.created_at
)
SELECT
    brand_id,
    count(*) AS direct_reply_pairs,
    count(DISTINCT customer_tweet_id) AS replied_customer_messages,
    median(date_diff('minute', customer_at, reply_at)) AS median_reply_minutes,
    quantile_cont(date_diff('minute', customer_at, reply_at), 0.90) AS p90_reply_minutes
FROM direct_pairs
GROUP BY brand_id
HAVING count(*) >= 8000
ORDER BY direct_reply_pairs DESC, brand_id ASC;
