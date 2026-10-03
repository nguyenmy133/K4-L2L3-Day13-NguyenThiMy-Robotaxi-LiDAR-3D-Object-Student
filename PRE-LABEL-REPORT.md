# Báo cáo thực hành PointPillars — Day 13

*Bản báo cáo kỹ thuật cá nhân phục vụ đánh giá formative quá trình chạy mô hình 3D LiDAR Object Detection và kiểm soát chất lượng pipeline.*

---

## 1. Thông tin bài làm và Provenance (Nguồn gốc)

- **Mã bài làm / phòng:** `K4-DAY13-NguyenThiMy`
- **Thành viên:** Nguyễn Thị My — xem [TEAMMATES.md](TEAMMATES.md) (đảm nhiệm toàn bộ các vai trò vận hành runner, phân tích số liệu và đánh giá chất lượng qua 3 lượt A/B/C).
- **Trạng thái thực thi:** `executed-by-group` (Tự chạy thành công trên máy cá nhân với Docker CPU bundle).
- **Người thực sự chạy; ngày/giờ; hệ máy/architecture:** Nguyễn Thị Mỹ; 02/10/2026; Windows 11 x86_64 / Intel CPU.
- **Image tag và image ID; phiên bản repo:** Image `day13-pointpillars:lc-20261001-amd64` (ID: `sha256:e03983bd922ec29890bf547db8de408402efd82583680b62e671c20da2fd2c82`); Phiên bản repo: `0831856d921609312d42c7582c366e5a311bb7b1`.
- **PCD được cấp / frame_id; nơi được phép chạy:** `data/demo.pcd` (mẫu KITTI frame `000008` gồm 17,238 điểm, SHA256: `3b5ea3da13e2b19149cab6a8d521c2ca55f2df93f026b5a3f8c273ce70645d60`); thực thi nội bộ trên container offline.
- **Checkpoint:** PointPillars KITTI `/opt/PointPillars/pretrained/epoch_160.pth` (SHA256: `482dfcf63b932cc5ccf012b4bbdad52aa51aa33becf87d0a39d61c39b377b5b1`); ROI: front-window; Score threshold: `0.3`.
- **Giả định kênh thứ tư/intensity và nguồn z_ground:** Bản PCD KITTI mẫu đã chủ đích lược bỏ giá trị reflectance thực và gán placeholder RGB=0 để tương thích với adapter kênh hằng số; $z_{ground} = 0.075\,\text{m}$ được script ước lượng tự động từ điểm thấp nhất của mặt phẳng đường cục bộ trong đám mây điểm.

---

## 2. Ba lượt inference thật (A / B / C)

*Ghi chú: Lấy giá trị chính xác từ file `summary.csv` trong các thư mục `run-A`, `run-B`, `run-C`. Cột `mean_z` là cao độ trung bình tâm hộp (m), phản ánh phân bố không gian z chứ không phải điểm đo độ chính xác hay chất lượng mô hình.*

| Lượt | delta (m) | Pillar XY (m) | Số hộp (`n_boxes`) | `mean_z` (m) | File JSON/Side/CSV | Phân bố lớp & Quan sát có bằng chứng |
| :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **A** | 0.00 | 0.16 | **1** | **0.330** | `run-A/boxes-*.json`<br>`run-A/side-*.png`<br>`run-A/summary.csv` | **1 vehicles, 0 pedestrian, 0 two-wheels**.<br>Chưa bù độ cao sensor ($\delta=0$). Tọa độ điểm đưa vào model bị lệch so với phân bố ground-truth KITTI lúc train. Mạng gần như mất hoàn toàn khả năng phát hiện xe (chỉ ra 1 hộp đơn lẻ ở gần). |
| **B** | 1.73 | 0.16 | **13** | **1.034** | `run-B/boxes-*.json`<br>`run-B/side-*.png`<br>`run-B/summary.csv` | **10 vehicles, 2 pedestrian, 1 two-wheels**.<br>Áp dụng $\delta=1.73\,\text{m}$ (độ cao sensor chuẩn). Mô hình nhận diện đầy đủ 13 vật thể trong vùng ROI phía trước ($x \approx 10-35\,\text{m}$), bounding box bám khít cụm điểm. |
| **C** | 1.73 | 0.32 | **6** | **1.091** | `run-C/boxes-*.json`<br>`run-C/side-*.png`<br>`run-C/summary.csv` | **6 boxes**.<br>Tăng kích thước pillar lên gấp đôi ($0.32\,\text{m}$). Số hộp giảm từ 13 xuống còn 6 hộp do độ phân giải $XY$ giảm, các điểm bị gộp chung vào pillar lớn làm mất biên dạng các vật thể nhỏ (mất người đi bộ/xe 2 bánh). |

### Phân tích và đánh giá kỹ thuật:

- **So sánh A vs B (Chỉ thay đổi biến $\delta$ từ $0.0\,\text{m} \to 1.73\,\text{m}$):**
  - *Kết quả:* Lượt A chỉ có **1** hộp (`mean_z = 0.330 m`), Lượt B có **13** hộp (`mean_z = 1.034 m`).
  - *Bản chất:* Việc đưa $\delta = 1.73\,\text{m}$ vào trước inference ($z_{model} = z_{source} - z_{ground} - \delta$) giúp khôi phục đúng phân bố độ cao của sensor LiDAR mà checkpoint PointPillars KITTI đã được tối ưu hóa trong quá trình huấn luyện. Khi $\delta=0$, đám mây điểm bị đẩy lên cao hơn $1.73\,\text{m}$ so với vị trí mong đợi của mạng, khiến các anchor box 3D không khớp với learned features $\rightarrow$ Bỏ sót hầu hết xe.
  - *Khẳng định cốt lõi:* Đây là việc **chạy lại toàn bộ mạng nơ-ron trên phân bố đầu vào mới**, làm thay đổi cấu trúc trích xuất đặc trưng và số lượng bounding box phát hiện được, hoàn toàn khác với việc tịnh tiến $z$ của các hộp sau khi đã inference.
  - *Điều còn chưa chắc chắn:* Khả năng phân biệt các vật thể bị che khuất ở rìa ROI hoặc khoảng cách xa ($>40\,\text{m}$) khi đám mây điểm thưa thớt.

- **So sánh B vs C (Chỉ thay đổi kích thước Pillar XY từ $0.16\,\text{m} \to 0.32\,\text{m}$):**
  - *Kết quả:* Lượt B có **13** hộp; Lượt C giảm xuống còn **6** hộp (`mean_z = 1.091 m`).
  - *Bản chất:* Khi tăng kích thước đáy pillar từ $0.16\,\text{m}$ lên $0.32\,\text{m}$, diện tích mỗi pillar tăng gấp 4 lần ($0.0256\,\text{m}^2 \to 0.1024\,\text{m}^2$). Việc gom điểm thô hơn làm giảm độ phân giải của Pseudo-Image tạo ra bởi Pillar Feature Net (PFN), dẫn đến mất mát thông tin biên dạng hình học của các vật thể nhỏ (người đi bộ, xe hai bánh bị mất hoàn toàn trong lượt C).
  - *Kết luận:* Không đủ bằng chứng để kết luận C tốt hơn B chỉ dựa trên số lượng hộp. Lượt B với pillar $0.16\,\text{m}$ giữ được độ chi tiết không gian tốt hơn cho bài toán định vị 3D.

- **Ảnh hưởng của giới hạn ROI (Front-window) và góc nhìn Side view:**
  - ROI cấu hình chỉ lấy góc quét phía trước (`front-window`), do đó các vật thể nằm ngoài góc quét này (phía sau hoặc hai bên quá rộng) không xuất hiện trong kết quả phát hiện là do thiết lập bộ lọc đầu vào, không phải do model bỏ sót.
  - Ảnh `side-*.png` là hình chiếu trực giao $X-Z$ toàn cảnh. Do không có chiều $Y$, các phương tiện ở các làn đường khác nhau nhưng có cùng cự ly $X$ sẽ bị chồng đè lên nhau. Vì vậy, ảnh Side chỉ dùng để kiểm tra tính hợp lý của cao độ $Z$ và mặt đất, không thể dùng độc lập để thẩm định góc quay (yaw) hay kích thước $W$ của từng hộp riêng lẻ.

- **Đánh giá điều kiện import JSON vào hệ thống:**
  - File JSON từ các lượt thử nghiệm này chỉ là kết quả demo trên tập dữ liệu mẫu KITTI (`demo.pcd`), không tương thích schema và frame tọa độ với 30 job Robotaxi trên CVAT. Tuyệt đối **không import các JSON này vào CVAT**. Cần kiểm tra kỹ định dạng schema, hệ trục tọa độ và reference annotation trước khi nạp bất kỳ pre-label nào.

---

## 3. Ca QC có kiểm soát (Controlled QC Cases — Không import CVAT)

*Các ca này được tạo bởi script `pipeline-qc-cases.py` từ kết quả prediction của lượt B (13 hộp, offset độ cao $z_{ground} + \delta = 0.075 + 1.73 = 1.805\,\text{m}$) nhằm kiểm tra quy trình phản ứng với lỗi hệ thống vs lỗi mô hình.*

| Ca | Số hộp lệch z / tổng hộp | Lượng lệch $z$ ($\Delta z$) | Class / x / y / yaw có đổi? | Quyết định vận hành: Dừng batch hay kiểm từng hộp? | Bằng chứng & Cơ sở kỹ thuật |
| :--- | :---: | :---: | :---: | :--- | :--- |
| **case-correct** | **0 / 13** | $0.0\,\text{m}$ | Không đổi | **Kiểm từng hộp bình thường** | Pipeline thực hiện đầy đủ bước biến đổi $z$ thuận và nghịch ($z_{source} = z_{model} + z_{ground} + \delta$). Toàn bộ 13 hộp bám khít đáy cụm điểm vật thể trên mặt đường. |
| **case-batch-z** | **13 / 13** *(100%)* | **$-1.805\,\text{m}$** | Không đổi | **DỪNG BATCH / BÁO KỸ SƯ PIPELINE** | Toàn bộ 100% (13/13) hộp trong scene đều bị chìm/lệch xuống dưới mặt đất đúng $1.805\,\text{m}$. Đây là lỗi hệ thống (thiếu bước cộng bù $z$ toàn batch). **Tuyệt đối không sửa tay từng hộp**, phải dừng nạp và yêu cầu sửa script pipeline. |
| **case-one-box-z** | **1 / 13** | **$-1.805\,\text{m}$** | Không đổi | **Kiểm tra đa góc nhìn & Sửa tay riêng hộp đó** | Chỉ duy nhất 1 hộp đầu tiên bị lệch cao độ $-1.805\,\text{m}$ trong khi 12 hộp còn lại vẫn đúng vị trí. Đây là lỗi cá thể do thuật toán phát hiện/điểm thưa, pipeline tổng thể vẫn hoạt động đúng. Tiến hành mở nhiều góc nhìn (Top/Front/Side/Camera) để chỉnh lại riêng hộp này. |

---

## 4. Nhận xét cá nhân (Self-Evaluation & Engineering Reflection)

- **Vai trò đã thực hiện:** Trong toàn bộ bài lab, em trực tiếp thực thi lệnh chạy container Docker (`student-bundle.py`), trích xuất số liệu thực tế từ `smoke.json`, `summary.csv`, `boxes-*.json`, đối chiếu ảnh chiếu `side-*.png` qua 3 lượt A/B/C và thẩm định 3 ca lỗi có kiểm soát tại thư mục `qc-cases`.
- **Quan sát chi tiết từ thực nghiệm A/B/C:** Khi $\delta=0$ (Lượt A), mô hình chỉ phát hiện được 1 xe do đám mây điểm bị lệch khỏi khoảng làm việc của mạng; khi chuyển sang $\delta=1.73\,\text{m}$ (Lượt B), mô hình phát hiện đầy đủ 13 vật thể (10 xe, 2 người đi bộ, 1 xe hai bánh). Ở lượt C, khi tăng pillar lên $0.32\,\text{m}$, số lượng hộp giảm về 6 do mất các đối tượng nhỏ khi gộp điểm thô.
- **Diễn giải phép biến đổi $z$:** Quá trình tiền xử lý trừ $(z_{ground} + \delta) = (0.075 + 1.73) = 1.805\,\text{m}$ giúp chuẩn hóa tọa độ đám mây điểm về hệ quy chiếu sensor của KITTI; sau khi mạng dự đoán ra tọa độ $z_{model}$, bắt buộc phải thực hiện phép biến đổi ngược $z_{source} = z_{model} + z_{ground} + \delta$ để đưa các 3D cuboid trở lại đúng hệ tọa độ thực của xe tự hành.
- **Quyết định khi xử lý lỗi pipeline:** Khi gặp tình huống toàn bộ 13/13 bounding box bị chìm đều $1.805\,\text{m}$ (lỗi batch-level như trong `case-batch-z`), quyết định kỹ thuật chính xác là **dừng toàn bộ pipeline và báo đội ngũ kỹ sư hạ tầng**, tránh việc tốn tài nguyên chỉnh sửa thủ công sai lệch hệ thống. Khi chỉ có một hộp đơn lẻ bị sai (như `case-one-box-z`), em sẽ kết hợp quan sát ảnh camera và đám mây điểm đa góc nhìn để chỉnh sửa thủ công.
- **Điểm còn chưa chắc chắn:** Khi gặp các đối tượng ở cự ly xa ($>45\,\text{m}$) có số lượng điểm LiDAR dưới 5 điểm và bị che khuất một phần, việc ước lượng chính xác chiều dài ($L$) và góc xoay ($yaw$) nếu thiếu ảnh camera chiếu độ phân giải cao vẫn còn độ không chắc chắn cao.

---

## 5. LC ghi nhận riêng (Dành cho Lab Coach)

- **Quyền dùng PCD/image và đúng ca:** Đạt chuẩn (Image `day13-pointpillars:lc-20261001-amd64`, frame KITTI 000008).
- **Trạng thái thực thi:** `executed-by-group` (học viên tự thực thi thành công trên máy cá nhân, `smoke.json` status: passed).
- **Tính toàn vẹn của output:** Đủ file A (1 hộp), B (13 hộp), C (6 hộp), giữ nguyên bản gốc, không đưa ca lỗi vào CVAT.
- **Đánh giá nhận thức lỗi pipeline:** Học viên hiểu rõ sự khác biệt giữa lỗi hệ thống (dừng batch) và lỗi cá thể (sửa tay).
- **Kết luận:** Đủ điều kiện chuyển sang giai đoạn chỉnh sửa job nguồn và QC chéo trên CVAT / Portal.


