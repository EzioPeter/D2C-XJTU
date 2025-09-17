import tyro
from dataclasses import dataclass

@dataclass
class Args:
    output_path: str = "output.txt"
    verbose: bool = False
    count: int = 3

def main(args: Args):
    print(f"输出路径: {args.output_path}")
    print(f"重复次数: {args.count}")
    if args.verbose:
        print("=== 详细模式 ===")
        print(f"正在处理文件: {args.input_path}")   
    print(f"处理完成，结果已保存到 {args.output_path}")

if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)    