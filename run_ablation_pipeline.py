import argparse

def main(argv=None):
    parser = argparse.ArgumentParser(description="FFXIV Crafter Sim Ablation Pipeline")
    parser.add_argument(
        "--target",
        default="all",
        help="アブレーションの対象 (デフォルト: all)",
    )
    args = parser.parse_args(argv)
    print(f"Running ablation for target: {args.target}")

if __name__ == "__main__":
    main()
