-- -- [CASE_1_MAIN] Slow Query Utama (Sesuai Skenario ReAct: Equality + Sort)
-- SELECT * FROM orders
-- WHERE customer_id = 1042
-- ORDER BY created_at DESC;

-- -- [CASE_2_ALT] Slow Query Alternatif (Filter Status + Pengurutan Nominal)
-- SELECT * FROM orders
-- WHERE status = 'COMPLETED'
-- ORDER BY total_amount DESC;

-- -- [CASE_3_GUARDRAIL] Uji Keamanan Input Guardrail (Perintah Destruktif)
-- DROP TABLE orders;

-- Uji Query JOIN: Menggabungkan orders (250k baris) dan customers (10k baris)
SELECT *
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
WHERE c.tier = 'VIP'
ORDER BY o.created_at DESC;