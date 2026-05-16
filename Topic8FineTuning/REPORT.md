# Topic 8 Fine-Tuning Report: Text-to-SQL with Tinker

## Summary

We fine-tuned `meta-llama/Llama-3.2-1B` with Tinker on a Text-to-SQL task using examples from `sql_create_context_v4.json`. The model was given a SQL table schema plus a natural-language question and trained to generate the corresponding SQL query.

The run used a focused 50-example training/evaluation slice so results could be collected within the workshop time window. Even on this small run, fine-tuning improved held-out accuracy from **50%** to **70%**.

## Run Setup

Command:

```bash
.venv/bin/python -u fine_tuning.py \
  --test-size 50 \
  --eval-limit 50 \
  --train-limit 50 \
  --batch-size 50 \
  --learning-rate 5e-4 \
  --epochs 1 \
  --show-errors 10 \
  --show-successes 5 \
  --model-name text-to-sql-50-example-run
```

Dataset fields:

- `question`: natural-language question
- `context`: SQL `CREATE TABLE` schema
- `answer`: expected SQL query

Prompt format:

```text
Table schema:
{context}
Question: {question}
SQL: {answer}
```

## Results

| Metric | Result |
|---|---:|
| Training examples | 50 |
| Held-out test examples | 50 |
| Base model accuracy | 50.00% |
| Fine-tuned accuracy | 70.00% |
| Absolute improvement | +20 percentage points |
| Relative improvement | +40% |
| Final training loss | 0.7226 |
| Average training loss | 0.7226 |
| Runtime | 781 seconds (~13 minutes) |

## Before vs. After

The fine-tuned model produced a clear improvement on the held-out Text-to-SQL examples. The base model already had some SQL knowledge, but fine-tuning made it more reliable at mapping schema fields and question phrasing into executable SQL.

The biggest observed gains were on direct schema-grounded lookups where the expected query required selecting the correct column and applying straightforward filters.

Example successes after fine-tuning:

### Success 1

Question:

```text
What is the lowest lap with a 145.926 qual?
```

Schema:

```sql
CREATE TABLE table_name_6 (laps INTEGER, qual VARCHAR)
```

Expected:

```sql
SELECT MIN(laps) FROM table_name_6 WHERE qual = "145.926"
```

Generated:

```sql
SELECT MIN(laps) FROM table_name_6 WHERE qual = "145.926"
```

### Success 2

Question:

```text
What is the outgoing manager when the date of vacancy is 10 october 2010?
```

Expected:

```sql
SELECT outgoing_manager FROM table_27683516_3 WHERE date_of_vacancy = "10 October 2010"
```

Generated:

```sql
SELECT outgoing_manager FROM table_27683516_3 WHERE date_of_vacancy = "10 october 2010"
```

This demonstrates successful schema grounding with harmless case variation.

### Success 3

Question:

```text
What batting partners batted for pakistan?
```

Expected:

```sql
SELECT batting_partners FROM table_1670921_2 WHERE batting_team = "Pakistan"
```

Generated:

```sql
SELECT batting_partners FROM table_1670921_2 WHERE batting_team = "Pakistan"
```

## Error Analysis

The fine-tuned model still made systematic mistakes. Most failures were valid-looking SQL with the wrong semantics rather than total syntax failures.

### 1. Missing aggregation

Question:

```text
what is the series # for the episode directed by Kelly Sandefur?
```

Expected:

```sql
SELECT MAX(no_in_series) FROM table_17901155_3 WHERE directed_by = "Kelly Sandefur"
```

Generated:

```sql
SELECT no_in_series FROM table_17901155_3 WHERE directed_by = "Kelly Sandefur"
```

Failure type: omitted `MAX()`.

Similar aggregation misses appeared with `COUNT`, `SUM`, and `MIN`.

### 2. Missing filter condition

Question:

```text
What is the score of Tim Herron, who placed t1?
```

Expected:

```sql
SELECT score FROM table_name_60 WHERE place = "t1" AND player = "tim herron"
```

Generated:

```sql
SELECT score FROM table_name_60 WHERE place = "t1"
```

Failure type: ignored one condition from the question.

### 3. Wrong column in filter

Question:

```text
How many drivers did Bob Gerard Racing have?
```

Expected:

```sql
SELECT COUNT(driver) FROM table_21977627_1 WHERE entrant = "Bob Gerard Racing"
```

Generated:

```sql
SELECT COUNT(driver) FROM table_21977627_1 WHERE driver = "Bob Gerard Racing"
```

Failure type: used the value with the wrong schema column.

### 4. Partial literal matching

Question:

```text
Name the result for new york 1
```

Expected:

```sql
SELECT result FROM table_1342249_32 WHERE district = "New York 1"
```

Generated:

```sql
SELECT result FROM table_1342249_32 WHERE district = "New York"
```

Failure type: dropped part of the literal value.

### 5. Join and multi-table reasoning errors

Question:

```text
Which campus has the most faculties in year 2003?
```

Expected:

```sql
SELECT T1.campus FROM campuses AS T1 JOIN faculty AS T2 ON T1.id = T2.campus WHERE T2.year = 2003 ORDER BY T2.faculty DESC LIMIT 1
```

Generated:

```sql
SELECT campus FROM campuses WHERE id = (SELECT MAX(id) FROM faculty WHERE year = 2003)
```

Failure type: attempted a subquery but did not preserve the intended join and ordering logic.

## Novel-Schema Questions

These examples used schemas outside the training distribution. The model did well on simple one-table lookups and struggled on counting, grouping, and joins.

| Scenario | Generated SQL | Judgment |
|---|---|---|
| Employee names in engineering | `SELECT name FROM employees WHERE department = "engineering"` | Correct |
| Count products over $50 | `SELECT name FROM products WHERE price > 50` | Incorrect: selected names instead of counting |
| Highest science score | `SELECT MAX(score) FROM students WHERE class = "science"` | Correct |
| Top 3 customers by total amount | `SELECT customer FROM orders ORDER BY amount DESC LIMIT 3` | Partial: missed `SUM(amount)` and `GROUP BY customer` |
| Students per department | `SELECT department FROM courses WHERE id IN (SELECT course_id FROM enrollments WHERE student_id = 1)` | Incorrect: missed count/grouping and join logic |

Novel-schema result: **2 correct, 1 partial, 2 incorrect**.

## RAG Comparison

A RAG system with 1,000 example question-SQL pairs would likely help when the new question is close to a retrieved example, especially for simple single-table patterns such as:

- selecting a column by equality filter
- counting rows with one simple condition
- using common aggregations like `MAX` or `MIN`

RAG would struggle when the schema or wording differs significantly from retrieved examples. The model still needs to compose a valid query using the current table names, column names, filters, joins, and aggregation logic. The observed failures highlight this: several incorrect outputs were structurally plausible but semantically wrong. Retrieval alone would not guarantee the model uses the right column, keeps all conditions, or builds the right join.

Fine-tuning appears to help the model internalize the task format and SQL-generation behavior, while RAG would mainly provide examples. For this task, examples are useful, but the model must still learn the compositional skill of translating schema-grounded language into executable SQL.

## Conclusion

The fine-tuned model improved from **50%** to **70%** accuracy on the held-out slice. It learned useful SQL syntax and schema grounding, especially for simple lookups and common aggregations. Remaining errors were mostly semantic: missing aggregations, missing filters, wrong columns, partial literal values, and multi-table reasoning mistakes.

Overall, the activity shows why fine-tuning is appropriate for Text-to-SQL: the target behavior is not just recalling facts, but repeatedly applying a structured transformation from schema plus question into SQL.
