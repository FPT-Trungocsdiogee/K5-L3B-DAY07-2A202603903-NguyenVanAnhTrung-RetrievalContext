# Code

| File | Công dụng |
|---|---|
| `retrieval.py` | `LexicalRetriever` (BM25), `DenseRetriever` (pgvector Day 6), `HybridRetriever` (RRF), `rrf_fuse`, `matches` (filter metadata) |
| `query_transform.py` | `QueryTransformer` (`none`, `rewrite`, `multi`, `hyde`, gọi model thật qua OpenRouter) và `TransformedRetriever` (chạy từng query rồi gộp bằng RRF) |
| `run_retrieval.py` | Chấm Hit@1, Evidence Hit@3, Context Evidence Rate theo split, ghi `traces.json` và `summary.json`, in bảng kết quả ra terminal |
| `show_results.py` | In lại kết quả đã lưu: so sánh nhiều lần chạy, bảng từng câu, chi tiết câu trượt |
| `run_agent.py` | Chạy agent thật (`run_agent` + `Controller`) trên một split, in trace từng bước, lưu `cases.json`/`traces.json`/`reviews.json` cho `agent_metrics.py` và in hai bảng agent của báo cáo dạng Markdown; `--show <thư mục>` in lại sau khi chấm nhãn |
| `report.py` | Hàm in bảng dùng chung cho hai script trên |

Mọi retriever có cùng hàm `search(query, *, filters, top_k)` như `agentic.Retriever`, nên dùng được trực tiếp với `run_agent`.

```bash
python src/run_retrieval.py --method lexical --split dev                    # -> starter/results/dev-lexical
python src/run_retrieval.py --method dense   --split dev                    # -> starter/results/dev-dense
  --transform multi --split dev  # -> starter/results/dev-hybrid-multi
python src/run_retrieval.py --method dense   --split dev --top-k 5 --output my_runs/dense-k5   # tự chọn thư mục
python src/run_retrieval.py --help   # top-k, budget, --use-filter, pool, rrf-k, trọng số, BM25 k1/b...
```

Mặc định `run_retrieval.py` in bảng từng câu (H@1, H@k, Ctx, trang gold, trang top-k với `*` là chunk chứa evidence, số chunk vào context) và phần tóm tắt. Thêm `--quiet` để chỉ in tóm tắt, `--misses` đểpython src/run_retrieval.py --method hybrid in evidence gold và đoạn text đã lấy cho các câu trượt, `--show-queries` để xem query sau transform, `--json` để in JSON thô.

Xem lại kết quả đã lưu mà không chạy lại:

```bash
python src/show_results.py "starter/results/dev-*"                  # bảng so sánh mọi lần chạy dev
python src/show_results.py starter/results/dev-dense --details      # bảng từng câu
python src/show_results.py starter/results/dev-dense --misses       # chi tiết các câu trượt
```

**Câu thiếu evidence (`agent_cases`):** `run_retrieval.py` đọc thêm khóa `agent_cases` trong [benchmark/benchmark.json](../benchmark/benchmark.json) (đổi bằng `--cases`, bỏ qua bằng `--no-agent-cases`). Chỉ lấy câu cùng split và `answerable=false`. Các câu này không có gold nên không tính Hit@k. Script chỉ chạy cùng retriever, ghi `absent_traces.json` (chunk lấy được, context, điểm top-1) và so điểm top-1 với các câu có đáp án. Cột `absent<min` cho biết bao nhiêu câu có điểm thấp hơn điểm top-1 thấp nhất của câu có đáp án, tức một ngưỡng điểm có thể loại được. Điểm chỉ so sánh được trong cùng một phương pháp và cấu hình. `absence_reason` không được gửi vào retrieval. Việc từ chối đúng hay sai vẫn phải chấm trên lần chạy agent.

`--output` không bắt buộc. Nếu bỏ trống, thư mục mặc định là `starter/results/<split>-<method>[-<transform>][-filter]`. Khi đổi tham số khác (top-k, budget, trọng số...), nên tự đặt `--output` để không ghi đè kết quả cũ.

Dense và hybrid cần Docker Day 6 cùng `DATABASE_URL`, `OPENROUTER_API_KEY`, `EMBEDDING_MODEL` trong `.env` (đọc `.env` gốc rồi `starter/.env`). Query transformation dùng `TRANSFORM_MODEL`, nếu không có thì `CHAT_MODEL`, mặc định `openai/gpt-4o-mini`. Kết quả transform được lưu trong `transform_cache.json` ở thư mục output để chạy lại không tốn thêm phí. Chỉ chạy `--split test` sau khi đã chốt cấu hình trên dev.
