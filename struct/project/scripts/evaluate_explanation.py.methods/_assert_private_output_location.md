> Source: `scripts/evaluate_explanation.py::_assert_private_output_location`

# `_assert_private_output_location(output_dir)`

Resolve symlink/path trước khi kiểm tra. Path ngoài repo được chấp nhận như private
storage do người dùng chỉ định. Path trong repo chỉ được chấp nhận khi
`git check-ignore --no-index` trả thành công; nếu Git vắng hoặc path không ignore
thì fail-closed. Trường hợp được phép vẫn phát privacy warning.

Risk: bỏ `resolve()` cho phép symlink né boundary; bỏ `--no-index` có thể làm
artifact đã lỡ track không còn được nhận diện đúng theo policy ignore.

← [`methods`](./_index.md)
