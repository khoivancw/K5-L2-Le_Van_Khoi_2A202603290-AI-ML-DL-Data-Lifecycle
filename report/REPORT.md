# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 11/06/2026

**Runtime Colab:** GPU T4

**Python / PyTorch / Ultralytics:** Python 3.12.10 / PyTorch 2.6.0+cu124 / Ultralytics 8.4.203

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35`

**Thay đổi so với notebook nguồn:** Không

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`=468 ; `class_name`=cab ; `rank`=1 ; `score`=0.510594 ; `taxonomy_name`=ImageNet-1K
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
   Record này nói về cả bức ảnh. Vì trong file json không có tọa độ vị trí và trong file chỉ có tên gán nhãn duy nhất(cab) và ma trận có kích thước 640*428pixel của bức ảnh.

3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
   Danh sách tên lớp đi kèm theo taxonomy / checkpoint (cụ thể là ImageNet-1K gồm 1.000 danh mục đã được định nghĩa từ trước khi huấn luyện). Mô hình không tự suy nghĩ hay sáng tạo ra tên lớp mới, mà chỉ tính toán xác suất phân loại trên 1.000 lớp cố định này.

4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp? 
   Cần ghi cả 3 vì phục vụ cho các đối tượng khác nhau:
   class_id: Cho máy tính đọc và tính toán bằng số nguyên cho nhanh và nhẹ bộ nhớ.
   class_name: Cho con người (annotator, reviewer) đọc để hiểu trực quan.
   taxonomy_name: Để biết đang dùng bộ từ điển nào. Vì mỗi bộ dữ liệu quy ước mã số khác nhau (ví dụ: cùng là số 5 nhưng bộ COCO là 'xe buýt', còn bộ Pascal VOC lại là 'cái chai'). Thiếu tên từ điển thì các hệ thống sẽ hiểu nhầm dữ liệu của nhau."

5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
   Guideline cần đưa ra luật chọn vật thể chính rõ ràng để mọi người làm giống nhau, cụ thể:
   Đo bằng diện tích: Vật nào to nhất, chiếm hơn nửa bức ảnh thì chọn vật đó.
   Đo bằng độ quan trọng: Nếu các vật to ngang nhau thì ưu tiên vật ở gần máy ảnh nhất, hoặc vật quan trọng nhất với bài toán (ví dụ: làm xe tự lái thì luôn ưu tiên chọn Người hơn là Ô tô hay Cây cối). -> Không có luật này thì 10 người gán nhãn sẽ cãi nhau và chọn 10 kiểu khác nhau.

6. Vì sao không được lấy `score` của model làm ground truth?
   Vì score chỉ là mức độ tự tin (xác suất đoán) của máy tính, mà máy tính thì hoàn toàn có thể đoán mò hoặc đoán sai. Còn ground truth là đáp án đúng thực tế ngoài đời do con người kiểm chứng. Nếu lấy phán đoán của máy làm đáp án chuẩn, máy sai cái gì mình cũng coi là đúng, hệ thống sẽ học theo cái sai đó và dẫn đến những lỗi nguy hiểm ngoài đời thực
## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`=person ; `score`=0.912615 ; `bbox_xyxy`=[385.34, 69.24, 498.92, 348.92] ; `bbox_width`=113.58 ; `bbox_height`=279.68

2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
   Hộp giới hạn bao quanh người đầu bếp bắt đầu từ góc trên-trái tại tọa độ pixel `(x=385.34, y=69.24)` và kéo dài đến góc dưới-phải tại tọa độ pixel `(x=498.92, y=348.92)`. Chiều rộng của hộp là `113.58 pixel` và chiều cao của hộp là `279.68 pixel`.

3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
     Ngưỡng `0.20`: có **17** predictions.
     Ngưỡng `0.60`: có **6** predictions.

4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
   - Khi hạ ngưỡng (ví dụ từ 0.60 xuống 0.20), số lượng hộp tăng vọt (từ 6 lên 17 hộp). Người reviewer sẽ phải tốn nhiều công sức để soi lọc và xóa bỏ các báo động giả (False Positive) do mô hình đoán mò vào các vùng nhiễu/nền, cũng như xử lý các hộp bị trùng lặp hoặc gán nhầm lớp.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
   Quy tắc Hộp ôm khít (Tight Box): Khi vẽ hộp cho bất kỳ đồ vật nào, người gán nhãn phải kéo 4 cạnh của hộp chạm sát vào các điểm mép ngoài cùng của vật thể đó.
   Không được vẽ hộp quá rộng: Không để thừa khoảng trống phông nền xung quanh.
   Không được cắt lẹm: Phải bao trọn toàn bộ đồ vật, không được cắt mất góc, mất chi tiết.
   Không bao trùm bóng: Chỉ vẽ thân vật thể, tuyệt đối không kéo hộp trùm lên bóng đen đổ dưới mặt đất/sàn nhà.

6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
   Người đầu bếp (person) bị quầy bếp che khuất mất phần chân từ hông trở xuống.
   Chiếc lò nướng (oven) ở góc bên trái bị rìa bức ảnh cắt cụt mất một phần thân bên trái. Chúng ta sẽ viết thành quy tắc trong Guideline, không cần hỏi Lab Coach (escalation). Vì hiện tượng đồ vật bị che khuất một phần hoặc nằm sát mép ảnh là chuyện rất bình thường và xuất hiện liên tục trong thực tế. Guideline cần ghi sẵn luật rõ ràng: "Vật nào bị che hoặc cắt mép thì cứ vẽ hộp ôm sát phần nhìn thấy được (Visible Box) và đánh dấu thuộc tính bị che khuất". Chỉ những ca dị biệt (ví dụ vật thể bị che khuất trên 90% không thể nhận ra là cái gì) thì mới cần đưa lên hỏi Lab Coach.

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`=kitchen-001 ; `class_name`=person ; `score`=0.899271 ; số điểm polygon=`polygon_point_count`=348 ; 3–5 điểm đầu của `polygon_xy`=[[446.0, 70.0], [445.0, 71.0], [444.0, 71.0], [443.0, 72.0], [442.0, 72.0]]

2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
So với Bounding Box (chỉ là cái khung hình chữ nhật thô và bị dính nhiều khoảng trống nền ở 4 góc), Polygon cung cấp thêm:
   Hình dạng và đường cong thực tế của vật thể (biết chính xác chỗ nào lồi, lõm, tay chân dang ra hay khép lại).
   Diện tích thực tế ở cấp độ từng điểm ảnh (pixel), loại bỏ hoàn toàn các pixel của phông nền xung quanh.
   Ranh giới tiếp xúc chính xác từng milimet giữa vật thể với các đồ vật bên cạnh."
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
   instance_id (ví dụ kitchen-001) dùng để phân biệt các cá thể riêng biệt cùng xuất hiện trong một bức ảnh tĩnh (ví dụ phân biệt người A và người B).
   Nó không phải class ID vì class ID dùng để chỉ chủng loại chung (cả 2 người đều chung class person).
   Nó không phải tracking ID vì nó chỉ có giá trị phân biệt trong một bức ảnh chụp tĩnh duy nhất, không dùng để theo vết chuyển động của đối tượng xuyên suốt thời gian qua các khung hình video."
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
   Đường biên đa giác (Polygon Mask) phải bám khít theo ranh giới quang học của vật thể; các góc bo và đường cong giải phẫu phải dùng mật độ điểm vừa đủ để không làm biến dạng hình khối tự nhiên; không vẽ lấn sang pixel của vật thể nền hoặc vật thể lân cận.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
     - **Vị trí:** Vùng tiếp xúc giữa đáy chiếc bát (`bowl`) và mặt bàn bếp, hoặc vị trí tay người cầm cán xoong/nồi.
     - Quyết định: Sử dụng Guideline
     - Vì: Đây là hiện tượng tiếp xúc vật lý thường gặp trong mọi căn bếp. Guideline cần có điều khoản quy định rõ: Tại ranh giới tiếp xúc hoặc đè lên nhau, vật thể ở lớp trên (foreground) được ưu tiên sở hữu các pixel đường biên, vật thể bị đè ở dưới sẽ kết thúc mask ngay tại mép cắt. Chỉ khi gặp vật liệu trong suốt/phản quang làm mờ nhòe hoàn toàn ranh giới quang học không thể phân biệt bằng mắt thì mới chuyển sang Escalation.

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh | 1 nhãn danh mục (string / category ID) gán cho toàn bộ bức ảnh. | Ảnh `traffic` có cả xe buýt, ô tô, người đi bộ nhưng model gán `cab` với score thấp (0.51). Khó chọn 1 nhãn bao quát. | Quan sát toàn ảnh, đối chiếu quy tắc vật thể nổi trội trong guideline để chọn 1 nhãn chuẩn xác nhất. | Kiểm tra nhãn được chọn có đúng quy tắc ưu tiên không; rà soát lỗi chọn sai ngữ cảnh toàn cục. |
| Phát hiện vật thể | Danh sách các hộp chữ nhật pixel `[x_min, y_min, x_max, y_max]` kèm `class_id` cho từng vật thể. | Các vật thể nhỏ ở xa (chai lọ trên kệ ảnh `kitchen`) hoặc vật thể bị che khuất dễ bị model bỏ sót ở ngưỡng cao (FN) hoặc vẽ hộp lỏng (loose box). | Vẽ hộp chữ nhật ôm sát từng vật thể nhìn thấy được (tight box); gắn nhãn lớp và cờ thuộc tính che khuất (`occluded`). | Soi độ khít của 4 cạnh hộp (IoU với vật thật); kiểm tra tỷ lệ sót vật thể (False Negative) và hộp thừa (False Positive). |
| Instance segmentation | Tập hợp tọa độ các đỉnh đa giác `[[x, y], ...]` kèm `class_id` và `instance_id` cho từng đối tượng. | Ranh giới giữa các đồ vật chạm nhau trên bàn bếp bị nhòe; đường viền tay người phức tạp dễ bị cắt phạm hoặc bao thừa nền. | Chấm các điểm neo (polygon points) ôm khít đường viền từng cá thể, gán đúng `instance_id` riêng biệt. | Kiểm tra độ mịn và độ chính xác của mask biên; đảm bảo không dính hai vật thể vào một mask và không để hở ranh giới tiếp xúc. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
   Chỉ sử dụng tập ảnh công khai COCO đã được cấp phép CC BY 2.0; tuyệt đối không đưa hình ảnh chứa khuôn mặt rõ nét, biển số xe cá nhân hoặc dữ liệu nội bộ của phương tiện lên các nền tảng AI/Cloud công cộng theo Luật Bảo vệ Dữ liệu Cá nhân Việt Nam.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
   - Dừng ngay lập tức việc xử lý/gán nhãn, không chia sẻ hay tải dữ liệu đó ra ngoài, và lập tức thông báo (escalate) cho **Giảng viên / Lab Coach phụ trách** để tiến hành cô lập và thu hồi dữ liệu.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
