import json

import pytest

from dais_sdk.tool.execute import _arguments_normalizer, _result_normalizer


class TestArgumentsNormalizer:
    """Test the _arguments_normalizer function that parses tool arguments"""

    def test_arguments_normalizer_string_to_dict(self):
        """JSON string should be parsed to dictionary"""
        result = _arguments_normalizer('{"name": "Alice", "age": 30}')
        assert result == {"name": "Alice", "age": 30}
        assert isinstance(result, dict)

    def test_arguments_normalizer_dict_passthrough(self):
        """Dictionary should pass through unchanged"""
        input_dict = {"key": "value", "count": 42}
        result = _arguments_normalizer(input_dict)
        assert result == input_dict
        assert result is input_dict  # Should be the same object

    def test_arguments_normalizer_empty_string(self):
        """Empty JSON object string should parse to empty dict"""
        result = _arguments_normalizer('{}')
        assert result == {}

    def test_arguments_normalizer_empty_dict(self):
        """Empty dict should pass through"""
        result = _arguments_normalizer({})
        assert result == {}

    def test_arguments_normalizer_nested_structure(self):
        """Nested JSON structures should be parsed correctly"""
        json_str = '{"user": {"name": "Bob", "roles": ["admin", "user"]}, "count": 5}'
        result = _arguments_normalizer(json_str)
        assert result == {
            "user": {"name": "Bob", "roles": ["admin", "user"]},
            "count": 5
        }

    def test_arguments_normalizer_chinese_characters_from_string(self):
        """Chinese characters in JSON string should be parsed correctly"""
        # JSON with Chinese characters
        result = _arguments_normalizer('{"message": "你好", "name": "张三"}')
        assert result == {"message": "你好", "name": "张三"}

        # JSON with Unicode escapes
        result = _arguments_normalizer('{"message": "\\u4f60\\u597d"}')
        assert result == {"message": "你好"}

    def test_arguments_normalizer_chinese_characters_from_dict(self):
        """Chinese characters in dict should pass through correctly"""
        input_dict = {"message": "你好世界", "user": "李四"}
        result = _arguments_normalizer(input_dict)
        assert result == input_dict

    def test_arguments_normalizer_various_types(self):
        """Various JSON types should be preserved"""
        json_str = '''
        {
            "string": "text",
            "number": 42,
            "float": 3.14,
            "boolean": true,
            "null": null,
            "array": [1, 2, 3],
            "object": {"nested": "value"}
        }
        '''
        result = _arguments_normalizer(json_str)
        assert result["string"] == "text"
        assert result["number"] == 42
        assert result["float"] == 3.14
        assert result["boolean"] is True
        assert result["null"] is None
        assert result["array"] == [1, 2, 3]
        assert result["object"] == {"nested": "value"}

    def test_arguments_normalizer_invalid_json(self):
        """Invalid JSON string should raise JSONDecodeError"""
        with pytest.raises(json.JSONDecodeError):
            _arguments_normalizer('invalid json')

        with pytest.raises(json.JSONDecodeError):
            _arguments_normalizer('{incomplete')


class TestResultNormalizer:
    """Test the _result_normalizer function that ensures all tool results are strings"""

    def test_result_normalizer_string_passthrough(self):
        """String results should pass through unchanged"""
        assert _result_normalizer("hello") == "hello"
        assert _result_normalizer("") == ""
        assert _result_normalizer("Hello, World!") == "Hello, World!"

    def test_result_normalizer_number_serialization(self):
        """Numbers should be converted to string representation"""
        assert _result_normalizer(42) == "42"
        assert _result_normalizer(3.14) == "3.14"
        assert _result_normalizer(0) == "0"
        assert _result_normalizer(-100) == "-100"

    def test_result_normalizer_boolean_serialization(self):
        """Booleans should be converted to JSON boolean strings"""
        assert _result_normalizer(True) == "true"
        assert _result_normalizer(False) == "false"

    def test_result_normalizer_none_serialization(self):
        """None should be converted to JSON null"""
        assert _result_normalizer(None) == "null"

    def test_result_normalizer_dict_serialization(self):
        """Dictionaries should be serialized to JSON"""
        result = _result_normalizer({"key": "value"})
        assert result == '{"key": "value"}'
        assert json.loads(result) == {"key": "value"}

        result = _result_normalizer({"a": 1, "b": 2})
        parsed = json.loads(result)
        assert parsed == {"a": 1, "b": 2}

    def test_result_normalizer_list_serialization(self):
        """Lists should be serialized to JSON"""
        result = _result_normalizer([1, 2, 3])
        assert result == "[1, 2, 3]"
        assert json.loads(result) == [1, 2, 3]

        result = _result_normalizer(["a", "b", "c"])
        assert json.loads(result) == ["a", "b", "c"]

    def test_result_normalizer_chinese_characters(self):
        """Chinese characters should be preserved (ensure_ascii=False)"""
        # String passthrough
        assert _result_normalizer("你好世界") == "你好世界"

        # Dict with Chinese
        result = _result_normalizer({"message": "你好", "name": "张三"})
        assert "你好" in result
        assert "张三" in result
        parsed = json.loads(result)
        assert parsed["message"] == "你好"
        assert parsed["name"] == "张三"

    def test_result_normalizer_nested_structures(self):
        """Nested data structures should be properly serialized"""
        data = {
            "user": {
                "name": "Alice",
                "age": 30,
                "roles": ["admin", "user"]
            },
            "active": True,
            "count": 5
        }
        result = _result_normalizer(data)
        parsed = json.loads(result)
        assert parsed == data

    def test_result_normalizer_empty_collections(self):
        """Empty collections should be serialized correctly"""
        assert _result_normalizer([]) == []
        assert _result_normalizer({}) == "{}"
        assert _result_normalizer(tuple()) == "[]"
