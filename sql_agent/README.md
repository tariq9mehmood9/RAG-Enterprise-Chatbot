## Quickstart

```sh
# An OPENAI_API_KEY is required
echo 'OPENAI_API_KEY=your_openai_api_key' >> .env

# A SQL database connection string is required
echo 'DATABASE_URL=sql_database_conn_string' >> .env

python src/run_service.py

# In another shell
streamlit run src/streamlit_app.py
```
