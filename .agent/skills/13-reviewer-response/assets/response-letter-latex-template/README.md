# 审稿回复 LaTeX 模板

恢复自仓库 2026-07-06 收录的六份模板文件。保留原有版式、署名、宏和分文件结构，仅整理了少量行末空白。模板是可选资产，不是当前论文，也不是每个新项目必须完成的任务。

## 使用

先确认工作目录不存在，再从仓库根目录复制。已有回复目录时继续编辑已有版本，不覆盖：

```bash
mkdir -p paper/reviews
# 仅当 paper/reviews/response-letter 尚不存在时执行。
cp -R .agent/skills/13-reviewer-response/assets/response-letter-latex-template paper/reviews/response-letter
cd paper/reviews/response-letter
latexmk -pdf -interaction=nonstopmode -halt-on-error review_response.tex
```

`review_response.tex` 填论文信息；`Reviewers/cover_letter.tex` 写已完成的修改摘要；`Reviewers/R1.tex` 至 `R3.tex` 分别填写原始意见和回应。按真实审稿人数增减工作副本，同时调整主文件的 `\input`。保留 `reviewresponse.sty` 及其相对路径。

编辑意见页可在工作副本中使用 `\editor`、`revcomment` 和 `revresponse`。`changes` 环境可引用已修改正文。原件中的 `\printpartbibliography` 宏需要另行配置 `biblatex`；默认示例未调用它。

Markdown 起草入口为 `paper/reviews/response_to_reviewers.md`。可直接写 LaTeX，无需同时维护两份回复。若从 Markdown 转写，明确后续以哪一份为准。

## 提交前

核对评论覆盖、真实修改、证据与位置，填写真实信息并清除工作副本中的 TODO，再编译和检查 PDF。模板原件可以保留 TODO；模板编译成功不代表内容可投稿。原始长标题占位可能产生排版警告，应在工作副本填写真实标题后处理。

不要在本资产目录写入真实审稿意见或提交编译缓存。保留 Karl-Ludwig Besser 与 Liwenhan Xie 的原署名；此次恢复不构成重新授权，不推断原件未列明的许可证。
