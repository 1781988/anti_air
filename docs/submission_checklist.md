# 比赛提交检查清单

- [ ] `python -m pytest -q` 全部通过；
- [ ] 数据检查无错误；
- [ ] 推理代码不读取文件名中的类别；
- [ ] 同一批次窗口未跨训练/验证折；
- [ ] `outputs/model/model.joblib` 存在；
- [ ] `run.sh` 可对任意一对 MAT/MP4 输出 JSON；
- [ ] `outputs/evaluation/metrics.json` 已生成；
- [ ] `outputs/report/test_report.md` 已生成；
- [ ] `docs/algorithm_design.md` 已纳入提交包；
- [ ] `outputs/submission/anti_air_submission.zip` 已在无网络环境解压测试；
- [ ] 服务器重新创建虚拟环境后可安装全部依赖；
- [ ] 正式评测接口、文件命名和输出字段与组委会最终说明一致。
