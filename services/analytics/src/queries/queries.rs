use deadpool_postgres::Pool;
use crate::models::models::{DayCount, DayStatusCount, LeaderboardEntry, UserSummary, TaskDetails};

type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub async fn get_lifetime_pages(pool: &Pool) -> Result<i64> {
    let client = pool.get().await.map_err(|e| {
        log::error!("Failed to get database connection: {:?}", e);
        e
    })?;
    
    let row = client.query_one(
        "SELECT COALESCE(SUM(COALESCE(page_count, 0)), 0) as total_pages FROM tasks",
        &[],
    ).await.map_err(|e| {
        log::error!("Database query failed: {:?}", e);
        e
    })?;
    
    Ok(row.get("total_pages"))
}

pub async fn get_pages_per_day(
    pool: &Pool,
    start: chrono::DateTime<chrono::Utc>,
    end: chrono::DateTime<chrono::Utc>,
    email: Option<&str>,
) -> Result<Vec<DayCount>> {
    let client = pool.get().await?;
    let query = if let Some(_email) = email {
        "SELECT date_trunc('day', t.created_at)::date AS day,
                COALESCE(SUM(t.page_count),0) AS pages
         FROM tasks t
         JOIN users u ON t.user_id = u.user_id
         WHERE t.created_at >= $1 AND t.created_at <= $2 AND u.email = $3
         GROUP BY 1
         ORDER BY 1;"
    } else {
        "SELECT date_trunc('day', created_at)::date AS day,
                COALESCE(SUM(page_count),0) AS pages
         FROM tasks
         WHERE created_at >= $1 AND created_at <= $2
         GROUP BY 1
         ORDER BY 1;"
    };
    let rows = if let Some(_email) = email {
        client.query(query, &[&start, &end, &_email]).await?
    } else {
        client.query(query, &[&start, &end]).await?
    };
    Ok(rows.iter().map(|r| DayCount {
        day: r.get("day"),
        pages: r.get("pages"),
    }).collect())
}

pub async fn get_status_breakdown(
    pool: &Pool,
    start: chrono::DateTime<chrono::Utc>,
    end: chrono::DateTime<chrono::Utc>,
    email: Option<&str>,
) -> Result<Vec<DayStatusCount>> {
    let client = pool.get().await?;
    let query = if let Some(_email) = email {
        "SELECT date_trunc('day', t.created_at)::date AS day,
                COALESCE(t.status, 'unknown') AS status,
                COALESCE(SUM(t.page_count),0) AS pages
         FROM tasks t
         JOIN users u ON t.user_id = u.user_id
         WHERE t.created_at >= $1 AND t.created_at <= $2 AND u.email = $3
         GROUP BY 1, t.status
         ORDER BY 1;"
    } else {
        "SELECT date_trunc('day', created_at)::date AS day,
                COALESCE(status, 'unknown') AS status,
                COALESCE(SUM(page_count),0) AS pages
         FROM tasks
         WHERE created_at >= $1 AND created_at <= $2
         GROUP BY 1, status
         ORDER BY 1;"
    };
    let rows = if let Some(_email) = email {
        client.query(query, &[&start, &end, &_email]).await?
    } else {
        client.query(query, &[&start, &end]).await?
    };
    Ok(rows.iter().map(|r| DayStatusCount {
        day: r.get("day"),
        status: r.get("status"),
        pages: r.get("pages"),
    }).collect())
}

pub async fn get_top_users(
    pool: &Pool,
    start: chrono::DateTime<chrono::Utc>,
    end: chrono::DateTime<chrono::Utc>,
    limit: i64,
) -> Result<Vec<LeaderboardEntry>> {
    let client = pool.get().await?;
    let rows = client.query(
        "SELECT COALESCE(u.email, 'unknown') AS email,
                COALESCE(SUM(t.page_count),0) AS total_pages
         FROM tasks t
         JOIN users u ON t.user_id = u.user_id
         WHERE t.created_at >= $1 AND t.created_at <= $2
         GROUP BY u.email
         ORDER BY total_pages DESC
         LIMIT $3;",
        &[&start, &end, &limit],
    ).await?;
    Ok(rows.iter().map(|r| LeaderboardEntry {
        email: r.get("email"),
        total_pages: r.get("total_pages"),
    }).collect())
}

pub async fn get_user_info(
    pool: &Pool,
    email: &str,
    start: chrono::DateTime<chrono::Utc>,
    end: chrono::DateTime<chrono::Utc>,
) -> Result<UserSummary> {
    let client = pool.get().await?;
    let row = client.query_one(
        "SELECT COALESCE(SUM(t.page_count),0)::int4 AS total_pages,
                COUNT(t.task_id)::int4 AS total_tasks,
                u.email,
                CONCAT(u.first_name, ' ', u.last_name) as name
         FROM users u
         LEFT JOIN tasks t ON t.user_id = u.user_id 
            AND t.created_at >= $2
            AND t.created_at <= $3
         WHERE u.email = $1
         GROUP BY u.email, u.first_name, u.last_name;",
        &[&email, &start, &end],
    ).await?;
    Ok(UserSummary {
        total_pages: row.get("total_pages"),
        total_tasks: row.get("total_tasks"),
        email: row.get("email"),
        name: row.get("name"),
    })
}

pub async fn get_task_details(
    pool: &Pool,
    start: chrono::DateTime<chrono::Utc>,
    end: chrono::DateTime<chrono::Utc>,
    email: Option<&str>,
) -> Result<Vec<TaskDetails>> {
    let client = pool.get().await?;
    let query = if let Some(_email) = email {
        "SELECT t.task_id, t.user_id, u.email, 
                CONCAT(u.first_name, ' ', u.last_name) as name,
                t.page_count, t.created_at, t.finished_at as completed_at,
                t.status
         FROM tasks t
         LEFT JOIN users u ON t.user_id = u.user_id
         WHERE t.created_at >= $1 AND t.created_at <= $2 AND u.email = $3
         ORDER BY t.created_at DESC
         LIMIT 100;"
    } else {
        "SELECT t.task_id, t.user_id, u.email, 
                CONCAT(u.first_name, ' ', u.last_name) as name,
                t.page_count, t.created_at, t.finished_at as completed_at,
                t.status
         FROM tasks t
         LEFT JOIN users u ON t.user_id = u.user_id
         WHERE t.created_at >= $1 AND t.created_at <= $2
         ORDER BY t.created_at DESC
         LIMIT 100;"
    };
    let rows = if let Some(_email) = email {
        client.query(query, &[&start, &end, &_email]).await?
    } else {
        client.query(query, &[&start, &end]).await?
    };
    Ok(rows.iter().map(|row| TaskDetails {
        task_id: row.get("task_id"),
        user_id: row.get("user_id"),
        email: row.get("email"),
        name: row.get("name"),
        page_count: row.get("page_count"),
        created_at: row.get("created_at"),
        completed_at: row.get("completed_at"),
        status: row.get("status"),
    }).collect())
}
