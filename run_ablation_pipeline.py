import argparse


def main(argv=None):
    """アブレーションパイプラインを実行するメイン関数。

    `--target` オプションを省略した場合でもエラーにならないように、
    デフォルト値を "all" に設定している。
    """

    parser = argparse.ArgumentParser(
        description="FFXIV Crafter Sim Ablation Pipeline"
    )
    parser.add_argument(
        "--target",
        default="all",
        required=False,
        help="アブレーションの対象 (デフォルト: all)",
    )
    args = parser.parse_args(argv)
    print(f"Running ablation for target: {args.target}")


if __name__ == "__main__":
    main()
