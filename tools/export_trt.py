import os
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('onnx_path', type=str)
    parser.add_argument('--trt_path', type=str, default=None)
    parser.add_argument('--precision', type=str, choices=['tf32', 'fp32'], default='tf32')
    return parser

def main(args):
    if not os.path.exists(args.onnx_path):
        raise FileNotFoundError(f"ONNX file not found: {args.onnx_path}")
    if args.trt_path is None:
        weight_folder = os.path.dirname(args.onnx_path)
        base_name = os.path.basename(args.onnx_path)
        trt_file_path = os.path.join(weight_folder, f'{os.path.splitext(base_name)[0]}_{args.precision}.engine')
    else:
        trt_file_path = args.trt_path
        trt_path = os.path.dirname(trt_file_path)
        os.makedirs(trt_path, exist_ok=True)

    command = f'trtexec --onnx={args.onnx_path} --saveEngine={trt_file_path}'

    # The fused opset-25 Attention nodes in DA3 have no dedicated kernel at
    # these shapes, so mark attention layers decomposable to fall back to
    # unfused attention (otherwise the build dies with a MyelinCheckException
    # "Attention operation was not supported by a dedicated kernel" and
    # produces an empty engine).
    command += " --decomposableAttentions='*'"

    if args.precision == 'fp32':
        command += ' --noTF32'

    os.system(command)

if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)