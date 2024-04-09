use axum::{routing::get, Router};
use tower_http::services::ServeDir;

#[tokio::main]
pub async fn run_registry() {
    // initialize tracing
    tracing_subscriber::fmt::init();

    // build our application with a route
    let app = Router::new()
        // `GET /` goes to `root`
        .route("/", get(root))
        .nest_service("/results", ServeDir::new("./results"));

    // run our app with hyper, listening globally on port 3000
    println!("Starting Cake registry...");
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}

// basic handler that responds with a static string
async fn root() -> &'static str {
    "Hello, World! This is the Cake registry speaking."
}
