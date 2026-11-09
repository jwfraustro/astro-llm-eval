# ADQL quick reference

ADQL (Astronomical Data Query Language) is the SQL dialect used by IVOA Table
Access Protocol (TAP) services. It is close to SQL-92 with a few differences
that trip up people coming from PostgreSQL or SQLite:

- Row limits use `SELECT TOP n ...`, not `LIMIT n`.
- Every TAP service exposes its own metadata in the `TAP_SCHEMA` schema:
  `TAP_SCHEMA.schemas`, `TAP_SCHEMA.tables`, `TAP_SCHEMA.columns`.
- Synchronous queries go to `{service}/sync` with the parameters
  `REQUEST=doQuery`, `LANG=ADQL`, `QUERY=...` and optionally `FORMAT=json`,
  `FORMAT=csv` or `FORMAT=votable`.
