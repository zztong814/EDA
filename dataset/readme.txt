
目录说明
1. 01_train_set:完整的模型训练数据
2. 02_public_test_set:目标工艺下各个问题的特征文件（features)
3. 03_submit_template:提交样例,统一提交格式：避免因文件名错误、列名缺失、格式不符导致的评分失败；  

使用流程
1. 从`01_train_set`获取训练数据，训练预测模型；  
2. 用模型对`02_public_test_set`的特征文件做预测，参考`03_submit_template`的文件名、格式要求,生成预测结果。  
