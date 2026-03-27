# 1. 修改视频url

对于json文件 `Task\05_adding_samples\Zuo_video_primary_day03.json` 中的视频url（"media_path"），将其替换为更新后的url，替换逻辑如下：

- 将url前缀统一改为：[https://huggingface.co/datasets/z4722/Implicit/blob/main/video/](https://huggingface.co/datasets/z4722/Implicit/blob/main/video/)
- 后缀统一采用local path中rawdata/后的部分，例如：`MUStARD/mmsd_raw_data/utterances_final/2_427.mp4`
- 最终形如：[https://huggingface.co/datasets/z4722/Implicit/blob/main/video/MUStARD/mmsd_raw_data/utterances_final/2_427.mp4](https://huggingface.co/datasets/z4722/Implicit/blob/main/video/MUStARD/mmsd_raw_data/utterances_final/2_427.mp4)

修改完后打开该url检验其是否存在，不存在的记录信息后汇总

# 2. 将csv文件中的值覆盖入json文件
 
对于csv文件 `Task\05_adding_samples\video_labels_Zuo_day03.csv` 中的内容，其与 `Task\05_adding_samples\Zuo_video_primary_day03.json` 中的内容一一对应，保持原json文件不变，将csv中的字段value回填导出到json文件 `Task\05_adding_samples\Zuo_video_primary_day03_export.json`

# 3. 修改json文件格式

对于导出后的json文件 `Task\05_adding_samples\Zuo_video_primary_day03_export.json` ，修改其格式与 `export\All_exported_labels_new_format.json` 一致

# 4. 生成subject与target混淆选项并规范格式

 - 新格式的json文件中，subject和target都各有3个混淆选项，且其格式都规范为小于等于5 words，同时不包含如斜杠`'/'`,`'\'`和括号`'('`,`')'`这样的标点符号，因此刚刚修改过来的json文件也应该遵守该格式
 - 注意，关于subject和target的混淆选项，一定要参考input text与audio caption以及视频内容来确认其准确性，混淆选项不能与真实选项指代相同，混淆选项最好为该视频场景下相关的内容
 - 例如，subject是`Amy`，而Amy穿着蓝色衬衫，这时如果混淆选项为`woman in blue`，该选项就与正确答案相同指代，就不能作为混淆选项