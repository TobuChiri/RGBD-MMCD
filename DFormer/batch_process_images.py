#!/usr/bin/env python3
"""
批量处理图片的特征图生成
这个脚本会多次调用单图片处理功能，避免内存泄漏问题
"""

import os
import sys
import time
import subprocess
import argparse
from datetime import datetime

def get_image_list(rgb_dir, start_id=0, end_id=99):
    """获取指定范围内的图片列表"""
    image_ids = []
    for i in range(start_id, end_id + 1):
        rgb_path = os.path.join(rgb_dir, f"{i}.jpg")
        if os.path.exists(rgb_path):
            image_ids.append(str(i))
    return image_ids

def run_single_image(image_id, script_path, rgb_dir, depth_dir, output_dir, log_dir):
    """运行单张图片处理"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"image_{image_id}_{timestamp}.log")

    print(f"\n{'='*80}")
    print(f"开始处理图片 {image_id}")
    print(f"日志文件: {log_file}")
    print(f"{'='*80}")

    # 构建命令
    cmd = [
        sys.executable, script_path,
        "--image_id", image_id,
        "--rgb_dir", rgb_dir,
        "--depth_dir", depth_dir,
        "--output_dir", output_dir
    ]

    start_time = time.time()

    try:
        with open(log_file, 'w') as f:
            # 运行命令并捕获输出
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )

            # 实时输出到日志文件和终端
            for line in process.stdout:
                f.write(line)
                f.flush()
                print(line, end='')

            # 等待进程完成
            return_code = process.wait()

        end_time = time.time()
        elapsed_time = end_time - start_time

        if return_code == 0:
            print(f"✓ 图片 {image_id} 处理成功，耗时: {elapsed_time:.2f}秒")
            return True, elapsed_time
        else:
            print(f"✗ 图片 {image_id} 处理失败，返回码: {return_code}，耗时: {elapsed_time:.2f}秒")
            return False, elapsed_time

    except Exception as e:
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"✗ 图片 {image_id} 执行异常: {e}，耗时: {elapsed_time:.2f}秒")
        with open(log_file, 'a') as f:
            f.write(f"\n执行异常: {e}\n")
        return False, elapsed_time

def main():
    parser = argparse.ArgumentParser(description='批量处理图片特征图')
    parser.add_argument('--start_id', type=int, default=0,
                       help='起始图片ID（包含）')
    parser.add_argument('--end_id', type=int, default=99,
                       help='结束图片ID（包含）')
    parser.add_argument('--batch_size', type=int, default=10,
                       help='每批处理的图片数量（0表示不分批）')
    parser.add_argument('--rgb_dir', type=str, default='datasets/NYUDepthv2/RGB/',
                       help='RGB图像目录')
    parser.add_argument('--depth_dir', type=str, default='datasets/NYUDepthv2/Depth/',
                       help='深度图像目录')
    parser.add_argument('--output_dir', type=str, default='feature_visual_output_separate_batch',
                       help='输出目录')
    parser.add_argument('--log_dir', type=str, default='batch_logs',
                       help='日志目录')

    args = parser.parse_args()

    # 创建日志目录
    os.makedirs(args.log_dir, exist_ok=True)

    # 获取图片列表
    image_ids = get_image_list(args.rgb_dir, args.start_id, args.end_id)

    if not image_ids:
        print(f"错误: 在目录 {args.rgb_dir} 中未找到ID {args.start_id} 到 {args.end_id} 的图片")
        return

    print(f"找到 {len(image_ids)} 张图片: {image_ids}")

    # 脚本路径
    script_path = os.path.join(os.path.dirname(__file__), 'analyze_all_channels_separate.py')

    # 分批处理
    if args.batch_size > 0:
        batches = [image_ids[i:i + args.batch_size] for i in range(0, len(image_ids), args.batch_size)]
    else:
        batches = [image_ids]

    total_success = 0
    total_failed = 0
    total_time = 0

    print(f"\n{'='*80}")
    print(f"开始批量处理，共 {len(batches)} 批")
    print(f"{'='*80}")

    for batch_idx, batch in enumerate(batches, 1):
        print(f"\n处理第 {batch_idx}/{len(batches)} 批，本批 {len(batch)} 张图片: {batch}")

        batch_start_time = time.time()
        batch_success = 0
        batch_failed = 0

        for image_id in batch:
            success, elapsed_time = run_single_image(
                image_id=image_id,
                script_path=script_path,
                rgb_dir=args.rgb_dir,
                depth_dir=args.depth_dir,
                output_dir=args.output_dir,
                log_dir=args.log_dir
            )

            if success:
                batch_success += 1
                total_success += 1
            else:
                batch_failed += 1
                total_failed += 1

            total_time += elapsed_time

            # 每处理完一张图片后等待一下，避免资源冲突
            time.sleep(1)

        batch_end_time = time.time()
        batch_elapsed = batch_end_time - batch_start_time

        print(f"第 {batch_idx} 批完成: 成功 {batch_success}, 失败 {batch_failed}, 耗时: {batch_elapsed:.2f}秒")

        # 每批之间等待一下
        if batch_idx < len(batches):
            print(f"等待5秒后处理下一批...")
            time.sleep(5)

    # 生成总结报告
    summary_file = os.path.join(args.log_dir, f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")
    with open(summary_file, 'w') as f:
        f.write(f"批量处理总结报告\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"{'='*60}\n")
        f.write(f"处理范围: ID {args.start_id} 到 {args.end_id}\n")
        f.write(f"找到图片: {len(image_ids)} 张\n")
        f.write(f"分批大小: {args.batch_size if args.batch_size > 0 else '不分批'}\n")
        f.write(f"RGB目录: {args.rgb_dir}\n")
        f.write(f"深度目录: {args.depth_dir}\n")
        f.write(f"输出目录: {args.output_dir}\n")
        f.write(f"日志目录: {args.log_dir}\n")
        f.write(f"{'='*60}\n")
        f.write(f"处理结果:\n")
        f.write(f"  成功: {total_success} 张\n")
        f.write(f"  失败: {total_failed} 张\n")
        f.write(f"  成功率: {total_success/(total_success+total_failed)*100:.1f}%\n")
        f.write(f"  总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)\n")
        f.write(f"  平均每张: {total_time/len(image_ids):.2f}秒\n")
        f.write(f"{'='*60}\n")
        f.write(f"处理的图片ID: {image_ids}\n")

    print(f"\n{'='*80}")
    print(f"批量处理完成！")
    print(f"成功: {total_success} 张，失败: {total_failed} 张")
    print(f"总耗时: {total_time:.2f}秒 ({total_time/60:.2f}分钟)")
    print(f"平均每张: {total_time/len(image_ids):.2f}秒")
    print(f"总结报告: {summary_file}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()