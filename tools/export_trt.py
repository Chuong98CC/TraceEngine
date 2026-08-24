import os
import argparse


def get_args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('onnx_path', type=str)
    parser.add_argument('--trt_path', type=str, default=None)
    parser.add_argument('--precision', type=str, choices=['fp16', 'tf32', 'fp32'], default='tf32')
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

    # TensorRT 11 removed --fp16 / --precisionConstraints / --layerPrecisions:
    # builds are strongly typed by the ONNX's own dtypes.  fp16 engines therefore
    # need a pre-cast fp16 ONNX (see cast_onnx_fp16.py).
    if args.precision == 'fp16' and not args.onnx_path.endswith('_fp16.onnx'):
        # scripts/export_trt_docker.sh pre-casts host-side and passes the casted
        # file; this branch covers running this script directly.
        from cast_onnx_fp16 import cast_to_fp16  # needs onnx + onnxconverter-common

        fp16_path = os.path.splitext(args.onnx_path)[0] + '_fp16.onnx'
        if not os.path.exists(fp16_path):
            print(f'Casting to fp16: {args.onnx_path} -> {fp16_path}')
            cast_to_fp16(args.onnx_path, fp16_path)
        args.onnx_path = fp16_path

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