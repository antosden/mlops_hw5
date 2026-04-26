docker run --name feast-postgres `
  -e POSTGRES_USER=tuser `
  -e POSTGRES_PASSWORD=12345 `
  -e POSTGRES_DB=postgres `
  -p 5433:5432 `
  -d postgres:15