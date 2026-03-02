from manual_tool_handling import calculator


def main():
    result = str(calculator("1+2"))
    print(result)
    assert result == "3"


if __name__ == "__main__":
    main()
