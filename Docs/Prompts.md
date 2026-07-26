make shortlist.sqlite:

## Goal
Rewrite the `db_creation` module
to create a new SQLite database named `shortlist.sqlite`
in the `db` folder using input data from `model_portfolios_shortlist_xls` (*.xls files).
## Context- The existing `model_portfolio.sqlite` database is correctly created
and must not be modified.- The new database `shortlist.sqlite`
should be generated from the input *.xls files located in `model_portfolios_shortlist_xls`.
- The import process for the new database should follow the same method as used
- for `model_portfolio.sqlite`:  - Use the same field names.
- Create "Date" fields in the same manner. 
- Avoid adding dated files already present in the database.## Output FormatProvide the rewritten
- `db_creation` module code or detailed instructions to implement the above requirements.
- <design_and_scope_constraints>- Do not modify the existing `model_portfolio.sqlite` database.
- The new database must be named `shortlist.sqlite` and placed in the `db` folder.
- Maintain consistency with the import method used in `model_portfolio.sqlite`.
</design_and_scope_constraints>

