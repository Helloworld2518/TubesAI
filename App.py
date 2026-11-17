import os
import re
import time
import sqlite3
from typing import TypedDict, List, Literal, Optional
from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Configurasi

OLLAMA_MODEL = "qwen2.5"
DB_PATH = "dummy_ecommerce.db"
SQL_FILE_PATH = "test_code.sql"

llm = ChatOllama(model=OLLAMA_MODEL, temperature=0)


# General Knowledge
def load_queries_from_sql_file(filepath: str = SQL_FILE_PATH) -> List[str]:
    """Membaca semua statement SQL dari test_code.sql secara bersih."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"File '{filepath}' tidak ditemukan! Buat file '{filepath}' berisi query SQL terlebih dahulu."
        )

    with open(filepath, "r", encoding="utf-8-sig") as f:
        raw_content = f.read()

    no_block_comments = re.sub(r"/\*.*?\*/", "", raw_content, flags=re.DOTALL)
    clean_lines = []
    for line in no_block_comments.splitlines():
        line_no_comment = re.sub(r"--.*$", "", line).strip()
        if line_no_comment:
            clean_lines.append(line_no_comment)

    joined_sql = " ".join(clean_lines)
    queries = [q.strip() + ";" for q in joined_sql.split(";") if q.strip()]
    return queries


KNOWLEDGE_BASE_DOCS = [
    {
        "doc_id": "KB-DB-001",
        "pattern_tags": ["where equality + order by sort", "composite index"],
        "rule": "Gunakan B-Tree Composite Index dengan urutan (equality_column, sort_column) untuk menghilangkan Sequential Scan dan Temp B-Tree Sort.",
        "example": "CREATE INDEX idx_table_eq_sort ON table_name(equality_col, sort_col DESC);"
    },
    {
        "doc_id": "KB-DB-002",
        "pattern_tags": ["where filter only", "single column index"],
        "rule": "Gunakan Single-Column B-Tree Index pada kolom filter WHERE untuk mengubah Full Table Scan menjadi Index Search.",
        "example": "CREATE INDEX idx_table_col ON table_name(filter_col);"
    },
    {
        "doc_id": "KB-DB-003",
        "pattern_tags": ["join foreign key + order by", "join foreign key", "nested loop join"],
        "rule": "Buat B-Tree Composite Index pada kolom Foreign Key tabel anak diikuti kolom pengurutan (fk_column, sort_column DESC) untuk mencegah Full Scan saat operasi JOIN.",
        "example": "CREATE INDEX idx_child_fk_sort ON child_table(foreign_key_col, sort_col DESC);"
    }
]


# Definisi Tools

def reset_custom_indexes(tables: List[str]):
    """Menghapus indeks buatan (idx_*) pada tabel terkait sebelum profiling awal."""
    if not os.path.exists(DB_PATH):
        return
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for tbl in tables:
        if not tbl:
            continue
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND tbl_name=? AND name LIKE 'idx_%';",
            (tbl,)
        )
        for (idx_name,) in cur.fetchall():
            cur.execute(f"DROP INDEX IF EXISTS {idx_name};")
    conn.commit()
    conn.close()


def get_table_columns(table_name: str) -> List[str]:
    """Mengambil daftar kolom asli dari tabel di SQLite."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(f"PRAGMA table_info({table_name});")
    cols = [row[1] for row in cur.fetchall()]
    conn.close()
    return cols


def get_explain_plan(sql_query: str, main_table: str = "orders") -> dict:
    """Tool Siklus 1 & 4: Mengeksekusi EXPLAIN QUERY PLAN dan mengukur waktu eksekusi nyata."""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(
            f"Database '{DB_PATH}' belum ada! Jalankan 'python setup_dummy_db.py' terlebih dahulu."
        )

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.execute(f"EXPLAIN QUERY PLAN {sql_query}")
    plan_rows = cur.fetchall()
    plan_text = " | ".join([row[3] for row in plan_rows])

    iterations = 20
    start_time = time.perf_counter()
    for _ in range(iterations):
        cur.execute(sql_query).fetchall()
    elapsed = (time.perf_counter() - start_time) / iterations

    cur.execute(f"SELECT COUNT(*) FROM {main_table}")
    total_rows = cur.fetchone()[0]
    conn.close()

    return {
        "plan": plan_text,
        "total_rows": total_rows,
        "execution_time_sec": round(elapsed, 4)
    }


def search_knowledge_base(search_query: str) -> dict:
    """Tool Siklus 2: Mencari aturan optimasi di Knowledge Base internal."""
    q_lower = search_query.lower().strip()
    for doc in KNOWLEDGE_BASE_DOCS:
        for tag in doc["pattern_tags"]:
            if tag in q_lower:
                return {
                    "status": "FOUND",
                    "doc_id": doc["doc_id"],
                    "rule": doc["rule"],
                    "example": doc["example"]
                }
    return {
        "status": "NOT_FOUND",
        "doc_id": None,
        "rule": "",
        "message": f"Tidak ditemukan aturan yang cocok untuk kueri: '{search_query}'"
    }


def generate_ddl_and_sql(raw_sql: str, parsed: dict) -> dict:
    """Tool Siklus 3: Menyusun DDL CREATE INDEX & refactoring SELECT * (mendukung JOIN & alias)."""
    main_table = parsed["table"]
    main_alias = parsed["table_alias"] or main_table
    join_table = parsed["join_table"]
    join_alias = parsed["join_alias"] or join_table
    fk_col = parsed["fk_col"]
    where_cols = parsed["where_cols"]
    sort_col = parsed["sort_col"]
    sort_dir = parsed["sort_dir"]

    # Output Guardrail: Ambil daftar kolom valid milik tabel utama agar tidak pernah salah kolom
    valid_main_cols = set(get_table_columns(main_table))

    index_cols = []
    name_parts = ["idx", main_table]

    # 1. Jika ada JOIN, kolom Foreign Key pada tabel utama berada di urutan pertama indeks
    if join_table and fk_col and fk_col in valid_main_cols:
        index_cols.append(fk_col)
        name_parts.append(fk_col)

    # 2. Tambahkan kolom WHERE yang benar-benar milik tabel utama
    for col in where_cols:
        if col in valid_main_cols and col not in index_cols:
            index_cols.append(col)
            name_parts.append(col)

    # 3. Tambahkan kolom ORDER BY jika milik tabel utama
    if sort_col and sort_col in valid_main_cols and sort_col not in index_cols:
        dir_suffix = f" {sort_dir}" if sort_dir == "DESC" else ""
        index_cols.append(f"{sort_col}{dir_suffix}")
        name_parts.append(sort_col)

    if not index_cols:
        fallback_col = "customer_id" if "customer_id" in valid_main_cols else next(iter(valid_main_cols))
        index_cols.append(fallback_col)
        name_parts.append(fallback_col)

    index_name = "_".join(name_parts)
    ddl = f"CREATE INDEX {index_name} ON {main_table}({', '.join(index_cols)});"

    # Refactor SELECT * menjadi daftar kolom eksplisit berawalan alias tabel
    main_cols = [c for c in get_table_columns(main_table) if c.lower() != "notes"]
    prefixed_cols = [f"{main_alias}.{c}" for c in main_cols]

    if join_table:
        join_cols = [c for c in get_table_columns(join_table) if c.lower() != fk_col]
        prefixed_cols.extend([f"{join_alias}.{c}" for c in join_cols])

    projection = ", ".join(prefixed_cols) if prefixed_cols else "*"
    refactored_sql = re.sub(
        r"^\s*SELECT\s+\*\s+FROM",
        f"SELECT {projection} FROM",
        raw_sql.strip(),
        flags=re.IGNORECASE
    )

    return {
        "index_name": index_name,
        "ddl_statement": ddl,
        "refactored_sql": refactored_sql
    }


def execute_db_ddl(ddl_statement: str) -> dict:
    """Tool Siklus 4: Mengeksekusi DDL ke SQLite database (dilindungi HITL)."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(ddl_statement)
    conn.commit()
    conn.close()
    return {
        "status": "SUCCESS",
        "executed_ddl": ddl_statement
    }


# 3. INPUT GUARDRAIL

def input_guardrail(sql_query: str) -> tuple[bool, str]:
    cleaned = sql_query.strip().upper()
    forbidden = ["DROP ", "DELETE ", "TRUNCATE ", "ALTER ", "UPDATE ", "INSERT "]
    if not cleaned.startswith("SELECT"):
        return False, "Input Guardrail Ditolak: Sistem hanya menerima query SELECT untuk dianalisis."
    for kw in forbidden:
        if kw in cleaned:
            return False, f"Input Guardrail Ditolak: Terdeteksi kata kunci berbahaya ({kw.strip()})."
    return True, "VALID"


# State

class AgentState(TypedDict):
    raw_sql: str
    next_worker: str
    step_count: int
    parsed_clauses: dict
    explain_before: dict
    rag_attempts: List[str]
    rag_status: str
    rag_rule: str
    index_name: str
    ddl_statement: str
    refactored_sql: str
    approval_status: str
    explain_after: dict
    final_answer: str


class SupervisorDecision(BaseModel):
    next_node: Literal[
        "query_parser_worker",
        "db_specialist_worker",
        "sql_refactor_worker",
        "execute_index_node",
        "FINISH"
    ] = Field(description="Pilih worker berikutnya berdasarkan tugas yang belum selesai.")
    reasoning: str = Field(description="Alasan singkat pemilihan routing.")


class RewrittenQuery(BaseModel):
    rewritten_query: str = Field(
        description="Kueri pencarian konseptual SQL untuk Knowledge Base."
    )


# Worker Nodes

def supervisor_node(state: AgentState) -> dict:
    step = state.get("step_count", 0) + 1
    if step > 10:
        return {"next_worker": "FINISH", "step_count": step}

    has_explain = bool(state.get("explain_before"))
    has_rag = state.get("rag_status") == "FOUND"
    has_ddl = bool(state.get("ddl_statement"))
    has_executed = bool(state.get("explain_after"))

    if not has_explain:
        expected_next = "query_parser_worker"
    elif not has_rag:
        expected_next = "db_specialist_worker"
    elif not has_ddl:
        expected_next = "sql_refactor_worker"
    elif not has_executed:
        expected_next = "execute_index_node"
    else:
        expected_next = "FINISH"

    prompt = (
        f"Tahapan berikutnya yang harus dijalankan adalah '{expected_next}'. "
        f"Kembalikan next_node='{expected_next}' beserta alasan singkat."
    )

    try:
        structured_llm = llm.with_structured_output(SupervisorDecision)
        decision = structured_llm.invoke([
            SystemMessage(content="Anda adalah Supervisor Router. Kembalikan JSON sesuai skema SupervisorDecision."),
            HumanMessage(content=prompt)
        ])
        chosen = decision.next_node if decision.next_node == expected_next else expected_next
    except Exception:
        chosen = expected_next

    print(f"\n [SUPERVISOR ROUTING - Step {step}] -> Mengarahkan ke: {chosen}")
    return {"next_worker": chosen, "step_count": step}


def query_parser_worker(state: AgentState) -> dict:
    sql = state["raw_sql"]
    reserved = {"JOIN", "INNER", "LEFT", "RIGHT", "WHERE", "ON", "ORDER", "GROUP", "LIMIT"}

    # 1. Ekstrak tabel utama & aliasnya (FROM orders o)
    from_match = re.search(r"\bFROM\s+(\w+)(?:\s+(?:AS\s+)?(\w+))?", sql, re.IGNORECASE)
    table_name = from_match.group(1) if from_match else "orders"
    raw_alias = from_match.group(2) if from_match else None
    table_alias = raw_alias if (raw_alias and raw_alias.upper() not in reserved) else None

    # 2. Ekstrak tabel JOIN, alias, & kondisi ON (JOIN customers c ON o.customer_id = c.customer_id)
    join_match = re.search(
        r"\bJOIN\s+(\w+)(?:\s+(?:AS\s+)?(\w+))?\s+ON\s+(?:(\w+)\.)?(\w+)\s*=\s*(?:(\w+)\.)?(\w+)",
        sql,
        re.IGNORECASE
    )
    join_table = None
    join_alias = None
    fk_col = None

    if join_match:
        join_table = join_match.group(1)
        j_alias = join_match.group(2)
        join_alias = j_alias if (j_alias and j_alias.upper() not in reserved) else None

        left_prefix, left_col = join_match.group(3), join_match.group(4)
        right_prefix, right_col = join_match.group(5), join_match.group(6)

        main_identifiers = {table_name.lower(), (table_alias or "").lower()}
        if left_prefix and left_prefix.lower() in main_identifiers:
            fk_col = left_col
        elif right_prefix and right_prefix.lower() in main_identifiers:
            fk_col = right_col
        else:
            fk_col = left_col

    # Bersihkan indeks lama pada tabel terlibat agar baseline akurat
    tables_to_reset = [table_name] + ([join_table] if join_table else [])
    reset_custom_indexes(tables_to_reset)

    # 3. Ekstrak kolom WHERE (pisahkan kolom tabel utama vs tabel JOIN)
    where_section = re.search(
        r"\bWHERE\s+(.*?)(?:GROUP\s+BY|ORDER\s+BY|LIMIT|;|$)",
        sql,
        re.IGNORECASE | re.DOTALL
    )
    main_where_cols = []
    all_where_cols = []
    if where_section:
        matches = re.findall(
            r"(?:(\w+)\.)?(\w+)\s*(?:=|!=|<>|>=|<=|>|<|\bLIKE\b|\bIN\b)",
            where_section.group(1),
            re.IGNORECASE
        )
        main_identifiers = {table_name.lower(), (table_alias or "").lower()}
        for prefix, col in matches:
            all_where_cols.append(f"{prefix}.{col}" if prefix else col)
            if not prefix or prefix.lower() in main_identifiers:
                main_where_cols.append(col)

    # 4. Ekstrak kolom ORDER BY (mendukung alias seperti o.created_at DESC)
    order_match = re.search(r"\bORDER\s+BY\s+(?:(\w+)\.)?(\w+)(?:\s+(ASC|DESC))?", sql, re.IGNORECASE)
    sort_col = order_match.group(2) if order_match else None
    sort_dir = (order_match.group(3) or "ASC").upper() if order_match else "ASC"


    """
    SIKLUS 1: Analisis EXPLAIN Plan & Profiling

    THOUGHT 1: Target awal adalah menganalisis slow query log. Saya harus mengeksekusi tool pengambil EXPLAIN plan untuk melihat letak bottleneck pada database.

    ACTION 1: 
    ExecuteTool(Tool="get_explain_plan", Input="SELECT * FROM orders WHERE customer_id = 1042 ORDER BY created_at DESC;")

    OBSERVATION 1: 
    Sequential Scan terdeteksi pada tabel orders (1.000.000 baris, waktu eksekusi: 4.2 detik).

    """

    print("SIKLUS 1: Analisis EXPLAIN Plan & Profiling (Query Parser Worker)")
    print(' THOUGHT 1:\n"Target awal adalah menganalisis slow query log. Saya harus mengeksekusi tool pengambil EXPLAIN plan untuk melihat letak bottleneck pada database."')
    print(f'\n ACTION 1:\nExecuteTool(Tool="get_explain_plan", Input="{sql}")')

    plan_result = get_explain_plan(sql, main_table=table_name)

    join_info = f" + JOIN {join_table} (FK: {fk_col})" if join_table else ""
    print(
        f"\n OBSERVATION 1:\n"
        f"Sequential Scan terdeteksi pada tabel {table_name}{join_info} ({plan_result['total_rows']:,} baris, "
        f"waktu eksekusi: {plan_result['execution_time_sec']} detik).\n"
        f"Detail Plan: [{plan_result['plan']}]"
    )

    return {
        "parsed_clauses": {
            "table": table_name,
            "table_alias": table_alias,
            "join_table": join_table,
            "join_alias": join_alias,
            "fk_col": fk_col,
            "where_cols": main_where_cols,
            "all_where_cols": all_where_cols,
            "sort_col": sort_col,
            "sort_dir": sort_dir
        },
        "explain_before": plan_result
    }


def db_specialist_worker(state: AgentState) -> dict:
    clauses = state["parsed_clauses"]
    join_table = clauses["join_table"]
    fk_col = clauses["fk_col"]
    where_cols = clauses["where_cols"]
    all_where = clauses["all_where_cols"]
    sort_col = clauses["sort_col"]
    attempts = []

    if join_table and sort_col:
        target_pattern = "JOIN foreign key + ORDER BY"
        desc_cols = f"JOIN {join_table} (FK {fk_col}), filter {', '.join(all_where)}, dan pengurutan {sort_col}"
    elif join_table:
        target_pattern = "JOIN foreign key"
        desc_cols = f"JOIN {join_table} pada kolom Foreign Key {fk_col}"
    elif where_cols and sort_col:
        target_pattern = "WHERE equality + ORDER BY sort"
        desc_cols = f"filter {', '.join(where_cols)} dan pengurutan {sort_col}"
    elif where_cols:
        target_pattern = "WHERE filter only"
        desc_cols = f"filter {', '.join(where_cols)}"
    else:
        target_pattern = "ORDER BY only"
        desc_cols = f"pengurutan {sort_col}"

    """
    SIKLUS 2: Pencarian Aturan di Knowledge 

    THOUGHT 2: 
    Sequential Scan terjadi pada kolom filter customer_id dan pengurutan created_at. 
    Saya perlu menggunakan tool pencarian memori/knowledge base untuk mencari aturan indexing.

    ACTION 2: 
    ExecuteTool(Tool="search_knowledge_base", Input="WHERE equality + ORDER BY sort")

    OBSERVATION 2: 
    Aturan ditemukan: 'Gunakan B-Tree Composite Index dengan urutan (equality_column, sort_column DESC)'.
    """
    print("SIKLUS 2: Pencarian Aturan di Knowledge Base (Database Specialist Worker)")

    print(
        f' THOUGHT 2:\n"Sequential Scan terjadi pada {desc_cols}. '
        f'Saya perlu menggunakan tool pencarian memori/knowledge base untuk mencari aturan indexing."'
    )

    initial_query = f"{clauses['table']} {join_table or ''} {fk_col or ''} {' '.join(all_where)} {sort_col or ''}".strip()
    initial_query = re.sub(r"\s+", " ", initial_query)
    attempts.append(initial_query)

    print(f'\n ACTION 2 (Percobaan 1):\nExecuteTool(Tool="search_knowledge_base", Input="{initial_query}")')
    res_1 = search_knowledge_base(initial_query)
    print(f"\n OBSERVATION 2A:\nStatus: {res_1['status']} -> {res_1.get('message')}")

    if res_1["status"] == "NOT_FOUND":
        print(
            f'\n SELF-CORRECTION (Query Rewriting):\n'
            f'"Pencarian awal NOT_FOUND. Melakukan Query Rewriting ke pola konseptual: {target_pattern}."'
        )
        try:
            rewriter = llm.with_structured_output(RewrittenQuery)
            rewritten = rewriter.invoke([
                SystemMessage(content=f"Ubah kueri SQL menjadi pola aturan '{target_pattern}'."),
                HumanMessage(content=f"Kueri gagal: {initial_query}")
            ])
            refined_query = rewritten.rewritten_query
            if target_pattern.lower() not in refined_query.lower():
                refined_query = target_pattern
        except Exception:
            refined_query = target_pattern

        attempts.append(refined_query)
        print(f'\n ACTION 2 (Percobaan 2 - Query Rewriting):\nExecuteTool(Tool="search_knowledge_base", Input="{refined_query}")')
        res_2 = search_knowledge_base(refined_query)
        print(f"\n OBSERVATION 2B:\nAturan ditemukan [{res_2['doc_id']}]: '{res_2['rule']}'")

        return {"rag_attempts": attempts, "rag_status": res_2["status"], "rag_rule": res_2["rule"]}

    return {"rag_attempts": attempts, "rag_status": res_1["status"], "rag_rule": res_1["rule"]}

"""
    SIKLUS 3: Refactoring SQL & Pembuatan Perintah DDL

    THOUGHT 3: 
    Berdasarkan hasil observasi knowledge base, saya akan menyusun DDL CREATE INDEX 
    dan merefaktor klausa SQL agar lebih efisien.

    ACTION 3: 
    ExecuteTool(Tool="generate_ddl_and_sql", Input="idx_orders_cust_date")

    OBSERVATION 3: 
    Draf DDL 'CREATE INDEX idx_orders_cust_date ON orders(customer_id, created_at DESC);' selesai dibuat.

"""

def sql_refactor_worker(state: AgentState) -> dict:
    clauses = state["parsed_clauses"]
    gen_result = generate_ddl_and_sql(raw_sql=state["raw_sql"], parsed=clauses)
    index_name = gen_result["index_name"]

    print("SIKLUS 3: Refactoring SQL & Pembuatan Perintah DDL (SQL Refactor Worker)")
    print(' THOUGHT 3:\n"Berdasarkan hasil observasi knowledge base, saya akan menyusun DDL CREATE INDEX dan merefaktor klausa SQL agar lebih efisien."')
    print(f'\n ACTION 3:\nExecuteTool(Tool="generate_ddl_and_sql", Input="{index_name}")')

    print(
        f"\n OBSERVATION 3:\n"
        f"Draf DDL '{gen_result['ddl_statement']}' selesai dibuat.\n"
        f"Refactored SQL: '{gen_result['refactored_sql']}'"
    )

    return {
        "index_name": index_name,
        "ddl_statement": gen_result["ddl_statement"],
        "refactored_sql": gen_result["refactored_sql"]
    }


"""
SIKLUS 4: Verifikasi & Eksekusi Akhir

THOUGHT 4: 
"Draf DDL sudah siap. Setelah dikonfirmasi, saya akan mengeksekusi DDL ke server database."

ACTION 4: 
ExecuteTool(Tool="execute_db_ddl", Input="CREATE INDEX idx_orders_cust_date...")

FINAL ANSWER: 
Indeks berhasil dibuat. Waktu eksekusi query turun dari x.x detik menjadi x.xx detik.
"""

def execute_index_node(state: AgentState) -> dict:
    ddl = state["ddl_statement"]
    index_name = state["index_name"]
    refactored_sql = state["refactored_sql"]
    table_name = state["parsed_clauses"]["table"]
    before_time = state["explain_before"]["execution_time_sec"]


    print("SIKLUS 4: Verifikasi & Eksekusi Akhir (Execute Index Node)")
  
    print(' THOUGHT 4:\n"Draf DDL sudah siap. Setelah dikonfirmasi, saya akan mengeksekusi DDL ke server database dan memverifikasi performanya."')
    print(f'\n ACTION 4:\nExecuteTool(Tool="execute_db_ddl", Input="{ddl}")')

    execute_db_ddl(ddl)
    after_plan = get_explain_plan(refactored_sql, main_table=table_name)
    after_time = after_plan["execution_time_sec"]

    # Evaluasi Pasca-Eksekusi & Auto-Rollback jika terjadi regresi waktu eksekusi
    if after_time >= before_time:
        rollback_ddl = f"DROP INDEX IF EXISTS {index_name};"
        execute_db_ddl(rollback_ddl)
        final_msg = (
            f"  REGRESI TERDETEKSI: Waktu eksekusi berubah dari {before_time} detik menjadi {after_time} detik "
            f"(tidak lebih cepat, kemungkinan karena selektivitas kolom rendah).\n"
            f"   • Self-Correction : Melakukan Auto-Rollback ({rollback_ddl}) untuk menjaga performa database."
        )
    else:
        final_msg = (
            f"Indeks berhasil dibuat. Waktu eksekusi query turun dari {before_time} detik "
            f"menjadi {after_time} detik.\n"
            f"   • DDL Dieksekusi : {ddl}\n"
            f"   • Plan Sebelum   : {state['explain_before']['plan']}\n"
            f"   • Plan Sesudah   : {after_plan['plan']}"
        )

    print(f"\n FINAL ANSWER:\n{final_msg}\n")

    return {
        "approval_status": "APPROVED",
        "explain_after": after_plan,
        "final_answer": final_msg
    }

# 6. LANGGRAPH BUILDER & RUNNER

def build_optimizer_graph():
    workflow = StateGraph(AgentState)
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("query_parser_worker", query_parser_worker)
    workflow.add_node("db_specialist_worker", db_specialist_worker)
    workflow.add_node("sql_refactor_worker", sql_refactor_worker)
    workflow.add_node("execute_index_node", execute_index_node)

    workflow.add_edge(START, "supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        lambda state: state["next_worker"],
        {
            "query_parser_worker": "query_parser_worker",
            "db_specialist_worker": "db_specialist_worker",
            "sql_refactor_worker": "sql_refactor_worker",
            "execute_index_node": "execute_index_node",
            "FINISH": END
        }
    )

    workflow.add_edge("query_parser_worker", "supervisor")
    workflow.add_edge("db_specialist_worker", "supervisor")
    workflow.add_edge("sql_refactor_worker", "supervisor")
    workflow.add_edge("execute_index_node", "supervisor")

    return workflow.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["execute_index_node"]
    )


def run_optimizer(sql_query: str, thread_id: str):
    print("*" * 75)
    print(f"MENJALANKAN QUERY DARI '{SQL_FILE_PATH}' (Model: {OLLAMA_MODEL})")
    print(f"Query SQL: {sql_query}")
    print("*" * 75)

    is_valid, msg = input_guardrail(sql_query)
    if not is_valid:
        print(f" [GUARDRAIL BLOCKED] {msg}\n")
        return

    app = build_optimizer_graph()
    config = {"configurable": {"thread_id": thread_id}}

    initial_state: AgentState = {
        "raw_sql": sql_query,
        "next_worker": "",
        "step_count": 0,
        "parsed_clauses": {},
        "explain_before": {},
        "rag_attempts": [],
        "rag_status": "",
        "rag_rule": "",
        "index_name": "",
        "ddl_statement": "",
        "refactored_sql": "",
        "approval_status": "PENDING",
        "explain_after": {},
        "final_answer": ""
    }

    app.invoke(initial_state, config)

    snapshot = app.get_state(config)
    if snapshot.next and "execute_index_node" in snapshot.next:
        vals = snapshot.values
        print(" HUMAN-IN-THE-LOOP (HITL) = Konfirmasi Dulu")
        print(f"Thread ID (MemorySaver) : {thread_id}")
        print(f"Node Ditahan            : {snapshot.next}")
        print(f"Draf DDL                : {vals['ddl_statement']}")
        print(f"Refactored SQL          : {vals['refactored_sql']}")
        print("-" * 75)

        choice = input("Setujui eksekusi DDL CREATE INDEX ke database? (y/n): ").strip().lower()
        if choice == "y":
            print("\n [DBA APPROVED] Melanjutkan dari state tersimpan...")
            app.invoke(None, config)
        else:
            app.update_state(config, {"approval_status": "REJECTED"})
            print("\n [DBA REJECTED] Eksekusi DDL dibatalkan.\n")


if __name__ == "__main__":
    queries = load_queries_from_sql_file(SQL_FILE_PATH)
    for idx, q in enumerate(queries, start=1):
        run_optimizer(q, thread_id=f"session-llama32-00{idx}")