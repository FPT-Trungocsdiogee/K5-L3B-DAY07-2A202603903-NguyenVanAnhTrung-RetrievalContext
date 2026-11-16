# Báo cáo Day 7

## Dữ liệu và benchmark

- Nguồn, quyền sử dụng, số trang:
 Phần 2 - Unit 1 Collecting and Preparing Data của giáo trình OpenStax Principles of Data Science (2025, bản PDF "WEB", ISBN 978-1-961584-60-0), tải từ https://openstax.org/details/books/principles-data-science ngày 2026-11-13. Giấy phép CC BY-NC-SA 4.0: chỉ dùng phi thương mại, giữ ghi công. PDF có 569 trang vật lý. Chỉ tính phần được nói đến(Introduction và các mục 2.1–2.5), tức PDF trang 67–103 (37 trang nội dung). Chi tiết nằm ở [data/manifest.json](../../data/manifest.json).
- Số chunk: 235 chunk, dùng lại index B của Day 6 (`by_paragraph(max_size=600)`: gom đoạn/câu liền nhau thành chunk ≤ 600 ký tự, không overlap; độ dài min 32, median 511, trung bình 479,6, max 600 ký tự; tổng 112.704 ký tự). Index A (`fixed_size(400, overlap=100)`, 381 chunk) vẫn nằm trong cùng bảng nhưng không dùng. Nguồn số liệu: `results/summary.json` của Day 6.
- Vector store, embedding model, phiên bản và số chiều: PostgreSQL 16.16 + pgvector 0.8.7 (Docker image `pgvector/pgvector:0.8.7-pg16`), bảng `chunks`, lọc `strategy = 'B'`. Metric cosine (`<=>`), tìm kiếm chính xác (không dùng index ANN); filter metadata chạy trong SQL. Embedding: `openai/text-embedding-3-small` qua OpenRouter, 1536 chiều. Tài liệu và câu hỏi dùng cùng model, không thêm tiền tố; [starter/agentic.py](../../starter/agentic.py) kiểm tra số chiều của vector câu hỏi so với cột `embedding` và báo lỗi nếu lệch, không có fallback sang vector giả. Phiên bản thư viện: Python 3.14.7, psycopg 3.3.8, pypdf 6.21.4.
- Agent model: `openai/gpt-4o-mini` qua OpenRouter, `temperature=0`, tool calling với `tool_choice="required"` (mỗi lượt model phải gọi `search` hoặc `answer`).
- Cách chạy và phân chia dev/test:
  - **Phân chia:** [benchmark/benchmark.json](../../benchmark/benchmark.json) giữ nguyên 21 câu có đáp án của Day 6, gồm **15 câu dev (q01–q15)** và **6 câu test (t01–t06)**. Mọi câu đều thuộc nguồn `openstax-pds-ch02`.
    - Dev: 5 `truc_tiep`, 4 `dien_dat_lai`, 3 `dieu_kien_ngoai_le`, 3 `metadata`. Ba câu metadata có filter `section` là q13 (2.3), q14 (2.5) và q15 (2.4).
    - Test: 2 `truc_tiep`, 1 `dien_dat_lai`, 1 `dieu_kien_ngoai_le`, 2 `metadata`. Hai câu metadata có filter là t05 (2.5) và t06 (2.4).
    - Câu thiếu evidence (`answerable=false`) nằm riêng trong khóa `agent_cases` để baseline retrieval không chấm chúng: 2 câu dev (a-q01, a-q02) và 2 câu test (a-t01, a-t02).
    - Mọi lần thử và chỉnh cấu hình chỉ chạy `--split dev`. Test chạy một lần sau khi đã chốt cấu hình, kết quả lưu vào thư mục riêng.
    - **Hạn chế:** báo cáo Day 6 ghi rằng tham số chunking đã được đổi sau khi xem kết quả test, và t06 được thêm vào sau các lần chạy trước. Vì vậy t01–t06 không còn là holdout độc lập hoàn toàn. Kết quả test ở Day 7 chỉ mang tính tham khảo cho tới khi có bộ test mới theo [GUIDE.md](../../GUIDE.md).
  - **Cách chạy** (từ thư mục gốc repo, Python ≥ 3.11, Docker Day 6 đang chạy, `.env` có `DATABASE_URL`, `OPENROUTER_API_KEY`, `EMBEDDING_MODEL`):
    ```bash
    python -m unittest discover -s tests -v                       # test code (dùng mock)
    python starter/check_day6_link.py --split dev --live          # kiểm tra model/manifest/evidence khớp Day 6
    python starter/export_day6.py --strategy B                    # xuất index B -> starter/data/day6/
    python starter/convert_day6.py --chunks starter/data/day6/chunks.json \
        --questions starter/data/day6/questions.json --output starter/data/converted
    python starter/baseline.py --chunks starter/data/converted/chunks.json \
        --questions starter/data/converted/questions.json --split dev --output starter/results/dev-tfidf
    # Các phương pháp retrieval (dense / BM25 / hybrid RRF, có thể thêm query transform):
    python src/run_retrieval.py --method hybrid --split dev --use-filter
    python src/show_results.py "starter/results/dev-*"            # bảng so sánh các lần chạy dev
    # Agent: chạy, in trace từng bước và bảng báo cáo; chấm reviews.json rồi in lại / tổng hợp:
    python src/run_agent.py --split dev --method hybrid --use-filter --budget 1600 \
        --output starter/results/agent-dev-hybrid-filter-citefix-b1600
    python src/run_agent.py --show results/agent-dev-hybrid-filter-citefix-b1600 --quiet
    python starter/agent_metrics.py --cases <thư mục>/cases.json --traces <thư mục>/traces.json \
        --reviews <thư mục>/reviews.json --split dev --output <thư mục>/metrics.json
    # Sau khi chốt cấu hình trên dev:
    python starter/baseline.py ... --split test --output starter/results/test-tfidf
    ```
    Dense retrieval truy vấn thẳng pgvector qua `Retriever` trong [starter/agentic.py](../../starter/agentic.py). Agent dùng `Controller` (OpenRouter, mặc định `openai/gpt-4o-mini`, đổi bằng `CHAT_MODEL`) và chạy qua `run_agent`; [src/run_agent.py](../../src/run_agent.py) bọc vòng lặp này để ghi trace và in bảng.
- Cách xác định câu có/thiếu evidence và tiêu chí chấm:
  - **Nhãn do người gán, không do hệ thống suy ra.** `answerable` là nhãn ground truth viết tay trong benchmark. Retriever và agent không bao giờ thấy nhãn này hay `absence_reason`; hai trường đó chỉ dùng khi chấm.
  - **Câu có evidence** (khóa `questions`): mỗi câu có `gold_source_id`, `gold_page` (số trang vật lý PDF, 67–103) và `evidence_quote` chép nguyên văn từ text trích xuất của trang gold. Nếu nhiều trang cùng hợp lệ thì ghi thêm `gold_alternatives`.
  - **Câu thiếu evidence** (khóa `agent_cases`, `answerable=false`, `expected_behavior="search_and_abstain"`): chọn chủ đề gần corpus (cùng từ vựng data science) nhưng nằm ngoài Chương 2, ví dụ learning rate, F1, p-value, random forest. Trước khi chốt, grep các từ khóa chính trên toàn bộ chunk để xác nhận không có kết quả; lý do và cách kiểm tra ghi trong `absence_reason`. Nếu grep tìm thấy evidence thì câu đó phải chuyển sang `questions` với gold đầy đủ.
  - **Tiêu chí chấm retrieval:** một chunk được tính là trúng khi khớp `source_id`, `page` và chứa `evidence_quote` dưới dạng chuỗi con nguyên văn (`hit` trong [starter/baseline.py](../../starter/baseline.py), giữ nguyên hoa/thường và xuống dòng; `convert_day6.py` giữ evidence đúng như text đã trích xuất). Riêng [starter/check_day6_link.py](../../starter/check_day6_link.py) kiểm tra trước, bằng so khớp đã bỏ khác biệt hoa/thường và khoảng trắng, rằng mỗi evidence có trong trang gold. Hit@1 tính trên chunk đầu tiên, Evidence Hit@3 tính trong top-3, còn Context Evidence Rate kiểm tra evidence có nằm trong context thực sự đưa cho agent hay không. Câu thiếu evidence không có gold nên không tính hit. Với nhóm này chỉ ghi lại kết quả retriever trả về và đếm số câu có điểm top-1 thấp hơn điểm top-1 thấp nhất của các câu có đáp án (`absent<min`), để xem một ngưỡng điểm có tách được hai nhóm hay không.
  - **Tiêu chí chấm agent** (nhãn audit thủ công, [starter/agent_metrics.py](../../starter/agent_metrics.py)):
    - Câu có đáp án được chấm `answer_grade` 0/1/2 (0 = sai, không có căn cứ hoặc từ chối; 1 = đúng một phần; 2 = đúng và có căn cứ đầy đủ). Từ chối trên câu có đáp án bắt buộc chấm 0 và tính là từ chối nhầm.
    - Câu thiếu evidence được chấm `abstention_correct`. Nhãn này chỉ đúng khi agent đã gọi search rồi mới từ chối, và không bịa đáp án.
    - Mỗi citation được gán một nhãn `citation_support` (true/false) cho biết chunk được trích có thực sự hỗ trợ câu trả lời hay không. Tính hợp lệ của citation không được suy ra từ việc ID có tồn tại.
    - Lỗi provider (`status="error"`) bị loại khỏi mẫu và báo cáo riêng.

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| Baseline | Dense (pgvector, index B, cosine), top-k=3, filter tắt, không rerank, budget 800 ký tự | Mốc so sánh: dense trên vector store Day 6, giữ nguyên cấu hình Day 6 | dev | 10/15 (67%) | 13/15 (87%) | 10/15 (67%) | 1234,2 ms/câu (median); $0,000005 cho 15 câu (embedding query) |
| Phương pháp 2 | Lexical BM25 (k1=1,5, b=0,75, trong bộ nhớ), top-k=3, filter tắt, không rerank, budget 800 | Câu hỏi chứa thuật ngữ chính xác (tên hàm, tên khái niệm) mà dense có thể xếp thấp; BM25 bắt khớp từ trực tiếp | dev | 8/15 (53%) | 14/15 (93%) | 8/15 (53%) | 0,3 ms/câu; $0 |
| Phương pháp 3 | Hybrid: dense + BM25, mỗi bên lấy pool 20, gộp bằng RRF (k=60, trọng số 1:1), top-k=3, filter tắt, budget 800 | Gộp hai danh sách để giữ Hit@1 của dense và độ phủ của BM25 | dev | 10/15 (67%) | 13/15 (87%) | 10/15 (67%) | 1402,9 ms/câu; $0,000005 |
| Cải thiện 1 | Hybrid như PP3, budget 1600 | Context Evidence Rate luôn bằng Hit@1 vì chunk dài tới 600 ký tự cộng header, nên budget 800 chỉ chứa 1 chunk. Nâng budget để chunk hạng 2–3 vào context | dev | 10/15 (67%) | 13/15 (87%) | 13/15 (87%) | 1339,6 ms/câu; $0,000005; context TB 1245 ký tự (2,1 chunk) |
| Cải thiện 2 | Hybrid, budget 1600, bật filter metadata `section` (chỉ q13–q15 có filter) | Filter giới hạn ứng viên trong đúng mục cho câu metadata, nhất là q15 (mục 2.4) | dev | 10/15 (67%) | 13/15 (87%) | 13/15 (87%) | 1370,5 ms/câu; $0,000005 |
| Cấu hình đã chọn | Hybrid RRF, top-k=3, filter bật, budget 1600 (= Cải thiện 2) | Chạy test một lần sau khi chốt trên dev | test | 4/6 (67%) | 6/6 (100%) | 6/6 (100%) | 1447,5 ms/câu; $0,000002 cho 6 câu |
| Baseline | Dense, top-k=3, filter tắt, budget 800 | Mốc trên test | test | 4/6 (67%) | 6/6 (100%) | 4/6 (67%) | 1357,6 ms/câu; $0,000002 |

Thời gian là median độ trễ retrieval mỗi câu (gồm gọi embedding qua mạng), đo trên máy cá nhân. Chi phí là tiền embedding OpenRouter báo cho cả split, không gồm agent. Cải thiện 1 và 2 không phải phương pháp mới; cả hai giữ phương pháp hybrid của PP3 và chỉ đổi một tham số mỗi lần. Kết quả từng lần chạy nằm trong [results/](../../results) (theo dõi bằng Git; `starter/results/` là thư mục làm việc và bị bỏ qua): `results/dense`, `results/lexical`, `results/hybrid` (Baseline, PP2, PP3), `results/hybrid-b1600` (Cải thiện 1), `results/hybrid-b1600-filter` (Cải thiện 2), `results/test-hybrid-b1600-filter` và `results/test-dense` (test), cùng các lần thử bên dưới. Xem lại bằng `python src/show_results.py results/dense results/lexical "results/hybrid*" "results/test-*"`.

Các lần thử khác trên dev, không chọn:
- `results/hybrid-multi`: hybrid + multi-query (gpt-4o-mini tạo 3 câu diễn đạt lại, gộp RRF), budget 800. Hit@1 giảm còn 6/15 (40%), Evidence Hit@3 13/15, Context Evidence Rate 6/15; 6382,6 ms/câu và $0,000466, tức đắt hơn khoảng 90 lần. Các câu diễn đạt lại làm loãng thứ hạng của chunk đúng.
- `results/lexical-filter`: BM25 có filter cho kết quả giống hệt BM25 không filter (8/15, 14/15, 8/15), vì không có filter BM25 cũng đã trả về chunk đúng mục cho q13, q14.
- `results/lexical-b1600-filter`: BM25, budget 1600, filter → 8/15, 14/15, 13/15. Context Evidence Rate bằng hybrid nhưng Hit@1 thấp hơn (8 so với 10).
- `results/hybrid-b1600-filter-wlex2`: hybrid, trọng số lexical 2 → 10/15, 13/15, 13/15, không đổi so với Cải thiện 2.

Hai lỗi trên dev, nguyên nhân, cách thử và lý do chọn cấu hình:
- **Lỗi 1, evidence đã lấy được nhưng không vào context (q02, q03, q04 với hybrid budget 800):** evidence nằm ở hạng 2 hoặc 3. `pack_context` giữ chunk hạng 1 (tới ~650 ký tự kể cả header) và bỏ chunk tiếp theo vì vượt 800 ký tự. Đây là lý do Context Evidence Rate trùng với Hit@1 ở mọi phương pháp chạy với budget 800. Nâng budget lên 1600 đưa 2–3 chunk vào context (TB 2,1 chunk, 1245 ký tự), và Context Evidence Rate tăng từ 10/15 lên 13/15, bằng Evidence Hit@3. Cùng lỗi này cũng làm agent từ chối nhầm (xem phần Agent).
- **Lỗi 2, chunk evidence xếp ngoài top-3 (q15, q05):** q15 hỏi "các loại missing data". Chunk `openstax-pds-ch02:p87:para600:0` kết thúc bằng câu "It is important to distinguish between different types of missing data" nên khớp rất mạnh và đứng đầu ở mọi lần chạy, trừ multi-query (ở đó `p87:para600:4` đứng đầu). Evidence lại nằm ở chunk kế tiếp (`p87:para600:1`, định nghĩa MCAR/MAR/MNAR), vốn dùng từ khác với câu hỏi. Đã thử bật filter `section=2.4`, BM25 riêng, tăng trọng số lexical và multi-query; không cách nào đưa chunk này vào top-3. q05 ("How many sampling techniques…") được BM25 và hybrid multi-query lấy trong top-3, nhưng hybrid một query thì không, kể cả khi trọng số lexical là 2. Multi-query bị loại vì Hit@1 tổng thể giảm từ 10/15 xuống 6/15. Chưa sửa được hai câu này. Hướng tiếp theo: gắn thêm chunk liền kề vào context (vì evidence hay nằm ngay sau câu dẫn), tăng top-k cùng budget, hoặc thêm reranker.
- **Lý do chọn hybrid + filter + budget 1600:** trên dev, ba cấu hình budget 1600 (hybrid, hybrid + filter, BM25 + filter) cùng đạt Context Evidence Rate 13/15. Hybrid giữ Hit@1 10/15 so với 8/15 của BM25, nên chunk hạng 1 (chunk luôn vào context đầu tiên) thường đúng hơn. Filter không tốn thêm chi phí, không làm giảm kết quả trên dev và giữ an toàn cho câu metadata. Đây cũng là cấu hình agent dùng (cấu hình C ở phần Agent).
- **So sánh trên test (chạy một lần sau khi chốt):** cấu hình đã chọn đạt Context Evidence Rate 6/6 so với 4/6 của baseline dense budget 800; Hit@1 bằng nhau (4/6), Evidence Hit@3 đều 6/6. Phần tăng thêm hoàn toàn đến từ budget, giống trên dev. Test chỉ có 6 câu (mỗi câu ≈ 17 điểm phần trăm) và không hoàn toàn độc lập (xem Hạn chế ở phần 1), nên chỉ mang tính tham khảo.
- **Đánh đổi:** budget gấp đôi làm context trung bình tăng từ ~580 lên ~1245 ký tự (prompt agent dài hơn) và đưa thêm chunk không liên quan vào context. Hybrid chậm hơn BM25 khoảng 1,3 s/câu do phải gọi embedding. Điểm RRF không dùng được làm ngưỡng để từ chối: trên test, điểm cosine của dense đặt cả 2 câu thiếu evidence dưới câu có đáp án thấp nhất (2/2), còn điểm RRF thì không tách được câu nào (0/2). Vì vậy quyết định từ chối để cho agent.

## Agent và đánh giá

- Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:
  - **Tool:** hai tool cho model: `search(query)` gọi retriever hybrid (dense pgvector index B + BM25, gộp bằng RRF, pool 20, k=60) với filter `section` của câu hỏi nếu có, top-k=3; `answer(answer, citations)` trả lời cuối. Model không thấy `answerable`, `absence_reason` hay gold.
  - **Số lượt gọi:** tối đa 4 lần `search` mỗi câu (tức tối đa 5 lượt model). Lần gọi `search` thứ 5 bị chặn.
  - **Điều kiện dừng:** (1) model gọi `answer` với citation hợp lệ → `answered`; (2) vượt giới hạn search → `abstain` (`search_limit`); (3) `answer` không có citation, hoặc citation không nằm trong context, hoặc ID không xuất hiện dạng `[chunk-id]` trong câu trả lời → `abstain` (`invalid_or_missing_citations`). Kiểm tra này chỉ xét ID, không xét nội dung có được hỗ trợ hay không; phần đó chấm thủ công.
  - **Chọn context:** context cộng dồn qua các lần search, `pack_context` giữ thứ tự (chunk của lần search trước đứng trước), bỏ chunk trùng và bỏ chunk không vừa budget. Model chỉ nhận context cộng dồn mới nhất và danh sách query đã chạy, không nhận lại toàn bộ lịch sử.
  - **Xử lý lỗi:** lỗi OpenRouter tạm thời (408/429/5xx/mạng) được thử lại tối đa 4 lần với backoff; nếu vẫn lỗi, câu đó được ghi `status="error"` và loại khỏi mẫu, không có kết quả giả. Trong ba lần chạy dưới đây không có lỗi provider nào.

Bốn lần chạy trên dev (15 câu có đáp án + 2 câu thiếu evidence). A → B → C mỗi lần chỉ đổi một yếu tố; B′ chạy lại đúng cấu hình B để kiểm tra độ ổn định:

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| A. Trước: hybrid+filter, k=3, budget=800, prompt gốc / dev | 17/17 (100%) | 32 search / 17 câu (TB 1,88); 49 lượt model | 0:15 1:0 2:0, TB 0,00 (bắt buộc 0 vì từ chối cả 15 câu) | n/a (0 citation) | n/a (0 câu trả lời) | _chưa chấm_ (2/2 đã search rồi từ chối) | 15/15 (100%) |
| B. A + sửa prompt citation / dev | 17/17 (100%) | 35 search / 17 câu (TB 2,06); 52 lượt model | _chưa chấm_ | _chưa chấm_ | 10/10 (100%) | _chưa chấm_ (2/2 đã search rồi từ chối) | 5/15 (33%) |
| B′. Chạy lại B / dev | 17/17 (100%) | 37 search / 17 câu (TB 2,18); 54 lượt model | _chưa chấm_ | _chưa chấm_ | 10/10 (100%) | _chưa chấm_ (2/2 đã search rồi từ chối) | 5/15 (33%) |
| C. B + budget=1600 (cấu hình đã chọn) / dev | 17/17 (100%) | 24 search / 17 câu (TB 1,41); 41 lượt model | _chưa chấm_ | _chưa chấm_ | 15/15 (100%) | _chưa chấm_ (2/2 đã search rồi từ chối) | 0/15 (0%) |

Bảng từng câu của cấu hình C (`*` = chunk chứa evidence gold, chấm tự động bằng `hit`; không thay cho nhãn thủ công; câu trả lời bị cắt ngắn, bản đầy đủ trong trace):

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| q01 | truc_tiep | search("Observational Data definition and collection methods") → p68*, p69, p68; ctx 2 chunk | answered: Observational data is defined as data collected by systematically observing and recording behavior without the active participation of th... — cite: openstax-pds-ch02:p68:para600:5 | chưa chấm |
| q02 | truc_tiep | search("Transactional data definition and collection methods") → p68, p68*, p69; ctx 2 chunk | answered: Transactional data refers to any type of information related to transactions or interactions between individuals, businesses, or systems.... — cite: openstax-pds-ch02:p68:para600:5 | chưa chấm |
| q03 | truc_tiep | search("Open-ended questions definition") → p71, p72*, p72; ctx 3 chunk | answered: Open-ended questions are designed to allow for more in-depth responses, providing respondents the opportunity to elaborate on their thoug... — cite: openstax-pds-ch02:p72:para600:0 | chưa chấm |
| q04 | truc_tiep | search("Close-ended questions definition") → p71, p72*, p72; ctx 3 chunk | answered: Close-ended questions are those that have predetermined answer choices, making them effective for gathering quantitative data. They are q... — cite: openstax-pds-ch02:p72:para600:0 | chưa chấm |
| q05 | truc_tiep | search("sampling techniques") → p72*, p72, p98; ctx 2 chunk | answered: The book mentions four sampling techniques: simple random selection, stratified sampling, cluster sampling, and convenience sampling [ope... — cite: openstax-pds-ch02:p72:para600:5 | chưa chấm |
| q06 | dien_dat_lai | search("importance of pre-testing a questionnaire on a small group") → p71*, p76, p75; ctx 2 chunk | answered: A questionnaire should be tried out on a small group before it is sent to everyone to identify any potential issues or confusion with the... — cite: openstax-pds-ch02:p71:para600:6 | chưa chấm |
| q07 | dien_dat_lai | search("how to compress a file without losing information") → p98*, p99, p98; ctx 2 chunk | answered: A file can be made smaller without losing any information through lossless compression techniques. Lossless compression reduces the size ... — cite: openstax-pds-ch02:p98:para600:5, openstax-pds-ch02:p99:para600:0 | chưa chấm |
| q08 | dien_dat_lai | search("how to see overall direction of stock index with volatile...") → p93*, p93, p93; ctx 2 chunk | answered: To see the overall direction of a stock index with volatile daily values, you can calculate a seven-day average, which helps smooth out f... — cite: openstax-pds-ch02:p93:para600:2, openstax-pds-ch02:p93:para600:4 | chưa chấm |
| q09 | dien_dat_lai | search("grouping exact ages into ranges reduce noise") → p93*, p93, p86; ctx 2 chunk | answered: The process of grouping exact ages into ranges, such as 0-10, 10-20, and 20-30, is called "binning". This technique helps to reduce noise... — cite: openstax-pds-ch02:p93:para600:3 | chưa chấm |
| q10 | dieu_kien_ngoai_le | search("systematic sampling biased sample situations") → p73*, p74, p73; ctx 2 chunk | answered: Systematic sampling can produce a biased sample if there is a pattern in the sampling frame, which is the organizing structure that repre... — cite: openstax-pds-ch02:p73:para600:6 | chưa chấm |
| q11 | dieu_kien_ngoai_le | search("outliers normalization formula") → p91*, p92, p91; ctx 3 chunk | answered: Before applying the normalization formula, it is important to remove any outliers from the dataset. This is necessary because outliers ca... — cite: openstax-pds-ch02:p91:para600:2 | chưa chấm |
| q12 | dieu_kien_ngoai_le | search("log transformation data conditions") → p92*, p92, p92; ctx 2 chunk | answered: A log transformation is typically applied to data when the data is highly skewed, meaning most of the data points fall toward one end of ... — cite: openstax-pds-ch02:p92:para600:3 | chưa chấm |
| q13 | metadata | search("pandas function reads HTML tables from a web page") → p78*, p78, p78; ctx 2 chunk | answered: The pandas function that reads HTML tables from a web page is `read_html()` [openstax-pds-ch02:p78:para600:2]. — cite: openstax-pds-ch02:p78:para600:2 | chưa chấm |
| q14 | metadata | search("splitting dataset into chunks fault tolerance") → p100*, p100, p100; ctx 2 chunk | answered: Splitting a dataset into chunks improves fault tolerance by allowing for the retrieval of only the affected chunk in case of data corrupt... — cite: openstax-pds-ch02:p100:para600:3 | chưa chấm |
| q15 | metadata | search("types of missing data in data science") → p87, p87, p87; ctx 2 chunk<br>search("types of missing data in data science") → p87, p87, p87; ctx 2 chunk | answered: In data science, it is important to distinguish between different types of missing data. The common types include: 1. **Missing Completel... — cite: openstax-pds-ch02:p87:para600:0 | chưa chấm |
| a-q01 | absence (thiếu evidence) | search("learning rate neural network cleaned dataset") → p93, p93, p92; ctx 2 chunk<br>search("recommended learning rate neural network") → p97, p98, p86; ctx 2 chunk<br>search("learning rate recommendation neural network") → p97, p77, p98; ctx 2 chunk<br>search("learning rate recommendation neural network training") → p97, p102, p98; ctx 2 chunk<br>search_limit | abstain: Không đủ bằng chứng để trả lời. — cite: [] | chưa chấm |
| a-q02 | absence (thiếu evidence) | search("F1 score formula classification model") → p91, p92, p91; ctx 3 chunk<br>search("F1 score formula") → p91, p92, p91; ctx 3 chunk<br>search("F1 score formula") → p91, p92, p91; ctx 3 chunk<br>search("F1 score formula") → p91, p92, p91; ctx 3 chunk<br>search_limit | abstain: Không đủ bằng chứng để trả lời. — cite: [] | chưa chấm |

- Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:
  - Trace (mỗi thư mục có `traces.json`, `cases.json`, `summary.json`, `reviews.json` để chấm): B [results/agent-dev-hybrid-filter-citefix](../../results/agent-dev-hybrid-filter-citefix), B′ [results/agent-dev-hybrid-filter](../../results/agent-dev-hybrid-filter), C [results/agent-dev-hybrid-filter-citefix-b1600](../../results/agent-dev-hybrid-filter-citefix-b1600). Trace của A từng nằm ở `agent-dev-hybrid-filter` nhưng đã bị ghi đè khi chạy B′. Số liệu của A ở trên lấy từ `summary.json`/`traces.json` gốc trước khi ghi đè. Muốn tái lập A thì bỏ phần sửa prompt citation trong `agentic.py` rồi chạy lại cấu hình B vào một thư mục mới.
  - Lệnh chạy: `python src/run_agent.py --split dev --method hybrid --use-filter [--budget 1600] --output <thư mục>`. Sau khi chấm `reviews.json`: `python src/run_agent.py --show <thư mục> --quiet` và `starter/agent_metrics.py` (xem phần Cách chạy).
  - Thời gian và chi phí (thời gian thực, chi phí do OpenRouter báo):

    | Lần chạy | Request model | Token (prompt + completion) | Chi phí | Thời gian/câu (median) | Tổng |
    |---|---|---|---|---|---|
    | A | 49 | 18.533 + 1.795 | $0,003861 | 5,5 s | 124,6 s |
    | B | 52 | 24.299 + 1.860 | $0,004765 | 5,4 s | 129,7 s |
    | B′ | 54 | 25.213 + 1.943 | $0,004952 | 6,5 s | 143,2 s |
    | C | 41 | 22.543 + 2.137 | $0,004667 | 5,1 s | 104,2 s |

    Chi phí embedding chỉ khoảng $0,000004 mỗi lần chạy và đã gồm trong các số trên.
- Kết quả trước/sau, ví dụ lỗi, đánh đổi và giới hạn:
  - **Lỗi 1, citation không nằm trong câu trả lời (A → B):** ở lần chạy A, model trả lời đúng ý và điền đúng chunk vào `citations` nhưng không viết `[chunk-id]` trong text, nên `run_agent` loại mọi câu trả lời và báo `abstain`. Ví dụ q01: câu trả lời đúng định nghĩa observational data và trích đúng `openstax-pds-ch02:p68:para600:5`, nhưng text không có ID nên bị chấm là từ chối. Sửa: system prompt và mô tả tham số `answer` trong [starter/agentic.py](../../starter/agentic.py) yêu cầu rõ ID trong ngoặc vuông ngay sau câu được hỗ trợ, kèm ví dụ. Không tự chèn ID vào câu trả lời sau khi model trả về. Từ chối nhầm giảm từ 15/15 xuống 5/15.
  - **Lỗi 2, context budget quá nhỏ (B → C):** chunk index B dài tới 600 ký tự, cộng header nên budget 800 chỉ chứa được 1 chunk. Khi evidence nằm ở hạng 2–3 (q02, q04, q15) thì không vào context; search lại cũng không giúp vì chunk của lần trước vẫn giữ chỗ, nên model lặp cùng một query tới khi chạm giới hạn (q04, q15 dùng đủ 4 search). Tăng budget lên 1600 (2–3 chunk): từ chối nhầm giảm từ 5/15 xuống 0/15, số search giảm từ 35 xuống 24, tổng thời gian giảm từ 129,7 s xuống 104,2 s, chi phí gần như không đổi vì mỗi prompt dài hơn nhưng số lượt ít hơn.
  - **Độ ổn định (B so với B′):** chạy lại cùng cấu hình B cho đúng cùng trạng thái ở cả 17 câu: cùng 10 câu được trả lời, mỗi câu trích cùng chunk, và cùng 5 câu từ chối nhầm (q02, q04, q05, q07, q15). Khác biệt duy nhất là q03 dùng 3 search thay vì 1, nên tổng số search tăng từ 35 lên 37. Vì vậy phần cải thiện A → B → C không phải do dao động ngẫu nhiên giữa các lần chạy, dù mới kiểm tra lặp lại cho B chứ chưa cho C.
  - **Điểm cần soi khi chấm cấu hình C:** q15 trả lời đủ MCAR/MAR/MNAR nhưng chunk evidence p87 không có trong kết quả retrieval; chunk được trích dừng ngay trước danh sách, nên phần định nghĩa có thể đến từ kiến thức sẵn của model chứ không từ nguồn. q02 trích đúng chunk có định nghĩa, nhưng phần "được thu thập thế nào" không có trong nguồn.
  - **Đánh đổi và giới hạn:** budget lớn hơn đưa thêm chunk không liên quan vào context, nên dễ trích nhầm hơn. Hai câu thiếu evidence vẫn được từ chối sau khi search (đúng hay không chờ nhãn thủ công), nhưng mỗi câu tốn đủ 4 lượt search và model lặp query (a-q02 gọi "F1 score formula" 3 lần). Kiểm tra citation trong `run_agent` chỉ xét ID nên không bắt được trường hợp như q15. Mới chạy trên dev, n=17, nên mỗi câu là gần 6 điểm phần trăm. `temperature=0` nhưng số lượt search vẫn có thể thay đổi giữa các lần chạy (q03 ở B/B′). Chưa chạy test.
